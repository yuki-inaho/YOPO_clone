# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from collections.abc import Mapping
import copy
from numbers import Real
import os
from typing import Optional, Sequence, Union
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle

import numpy as np
from mmcv.ops import batched_nms
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
import torch

from yopo.registry import METRICS
from ..functional import eval_map
from .oriented_box_iou_3d import (
    oriented_box_iou_3d,
    pairwise_oriented_box_iou_upper_bound_from_boxes_3d,
)


_COCO_IOU_THRESHOLDS = tuple(np.arange(0.50, 0.96, 0.05).tolist())


def normalize_2d_iou_thresholds(
    iou_thrs: Union[Real, Sequence[float]],
) -> tuple[float, ...]:
    """Validate and normalize configured 2D AP IoU thresholds.

    Duplicate thresholds are rejected instead of silently giving them extra
    weight in the mean AP. The user-provided order is retained for logging.
    """
    if isinstance(iou_thrs, bool):
        raise TypeError("iou_thrs must be a number or a sequence of numbers")
    if isinstance(iou_thrs, Real):
        values = (float(iou_thrs),)
    else:
        if isinstance(iou_thrs, (str, bytes)):
            raise TypeError("iou_thrs must be a number or a sequence of numbers")
        try:
            values = tuple(float(value) for value in iou_thrs)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "iou_thrs must be a number or a sequence of numbers") from error

    if not values:
        raise ValueError("iou_thrs must contain at least one threshold")
    if not np.isfinite(values).all():
        raise ValueError(f"iou_thrs must contain only finite values, got {values}")
    if any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError(
            f"iou_thrs values must be in the interval (0, 1], got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"iou_thrs must not contain duplicates, got {values}")
    return values


def _ap_key(iou_thr: float) -> str:
    """Return an unambiguous metric key for an IoU threshold."""
    percentage = iou_thr * 100.0
    if np.isclose(percentage, round(percentage), rtol=0.0, atol=1e-9):
        suffix = str(int(round(percentage)))
    else:
        suffix = f"{percentage:.12g}".replace(".", "p")
    return f"AP{suffix}"


def _is_coco_iou_range(iou_thrs: Sequence[float]) -> bool:
    """Whether thresholds are exactly COCO's AP50:95 ten-point range."""
    return len(iou_thrs) == len(_COCO_IOU_THRESHOLDS) and np.allclose(
        sorted(iou_thrs), _COCO_IOU_THRESHOLDS, rtol=0.0, atol=1e-9)


def resolve_hbb_selection(
    hbb_selection: Optional[Mapping],
    *,
    default_score_thr: float,
    default_nms_cfg: Optional[dict],
) -> Optional[dict]:
    """Resolve an optional 2D-only selection policy.

    ``None`` deliberately means "reuse the normal 2D/3D aligned prediction",
    preserving the historical behavior without storing a duplicate result.
    """
    if hbb_selection is None:
        return None
    if not isinstance(hbb_selection, Mapping):
        raise TypeError("hbb_selection must be a mapping or None")
    unknown_keys = set(hbb_selection) - {"score_thr", "nms_cfg"}
    if unknown_keys:
        raise ValueError(
            f"unsupported hbb_selection keys: {sorted(unknown_keys)}")
    try:
        score_thr = float(hbb_selection.get("score_thr", default_score_thr))
    except (TypeError, ValueError) as error:
        raise TypeError("hbb_selection.score_thr must be a number") from error
    if not np.isfinite(score_thr):
        raise ValueError("hbb_selection.score_thr must be finite")
    return {
        "score_thr": score_thr,
        "nms_cfg": hbb_selection.get("nms_cfg", default_nms_cfg),
    }


def select_aligned_prediction_indices(
    pred,
    *,
    score_thr: float,
    nms_cfg: Optional[dict],
) -> torch.Tensor:
    """Select query indices once so every 2D/3D field stays aligned."""
    scores = pred["scores"]
    keep = torch.arange(len(scores), device=scores.device)
    if score_thr > 0:
        keep = keep[scores > score_thr]
    if nms_cfg is not None and keep.numel():
        _, local_keep = batched_nms(
            pred["bboxes"][keep],
            scores[keep],
            pred["labels"][keep],
            nms_cfg,
        )
        keep = keep[local_keep]
    return keep


def rescale_ground_truth_bboxes(
    bboxes: np.ndarray,
    scale_factor,
) -> np.ndarray:
    """Map pipeline-resized GT boxes back to prediction/original coordinates.

    YOPO's detector ``predict`` contract returns rescaled boxes in the original
    image coordinate system.  Validation annotations, however, remain in the
    resized pipeline coordinate system.  Metrics must undo that resize before
    comparing the two.  ``scale_factor`` follows the MMDetection ``(w, h)``
    convention and may also be supplied as ``(w, h, w, h)``.
    """
    bboxes = np.asarray(bboxes).copy()
    if scale_factor is None or bboxes.size == 0:
        return bboxes
    scale = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    if scale.size == 2:
        scale = np.tile(scale, 2)
    if scale.size != 4 or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(
            "scale_factor must contain positive finite (w, h) or "
            f"(w, h, w, h) values, got {scale_factor!r}")
    return bboxes / scale.astype(bboxes.dtype, copy=False)


@METRICS.register_module()
class NOCSMetric(BaseMetric):
    """NOCS evaluation metric.

    ``3d_iou_*`` uses the exact intersection volume of arbitrarily oriented
    SO(3) cuboids.  Results produced before the 2026-08-26 correction used a
    corner-axis reduction and are not numerically comparable.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'mAP', 'recall'. If is list, the first setting in the list will
             be used to evaluate metric.
        models_path (str): Path to the models used for evaluation.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        iou_thrs (float | Sequence[float]): 2D bbox IoU thresholds passed to
            :func:`eval_map`. The default ``0.5`` preserves the historical
            ``AP50``-only behavior. Configuring ``0.50:0.05:0.95`` additionally
            reports their arithmetic mean as ``AP50_95``.
        hbb_selection (dict, optional): Optional 2D AP-only prediction
            selection with ``score_thr`` and ``nms_cfg`` keys. When omitted,
            2D AP and 3D pose use the same historical selection. For an
            unfiltered diagnostic sweep, set ``dict(score_thr=0.0,
            nms_cfg=None)``; 3D pose and dumped predictions remain unchanged.
        two_phase_3d_iou (bool): Use proof-safe upper-bound pruning before the
            exact 3D OBB kernel. Defaults to True. Set False to A/B runtime or
            retain every below-threshold diagnostic overlap value.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(
        self,
        format_only: bool = False,
        collect_device: str = "cpu",
        nms_cfg: dict = dict(type="nms", iou_threshold=0.5),
        score_thr: float = 0.2,
        dump_results_path: Optional[str] = None,
        dump_format: str = "pickle",  # "pickle" or "json"
        prefix: Optional[str] = None,
        iou_thrs: Union[Real, Sequence[float]] = 0.5,
        hbb_selection: Optional[dict] = None,
        two_phase_3d_iou: bool = True,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.nms_cfg = nms_cfg
        self.score_thr = score_thr
        self.iou_thrs = normalize_2d_iou_thresholds(iou_thrs)
        self.hbb_selection = resolve_hbb_selection(
            hbb_selection,
            default_score_thr=score_thr,
            default_nms_cfg=nms_cfg,
        )
        if not isinstance(two_phase_3d_iou, bool):
            raise TypeError("two_phase_3d_iou must be a bool")
        self.two_phase_3d_iou = two_phase_3d_iou
        self.format_only = format_only
        self.dump_results_path = dump_results_path
        assert dump_format in ["pickle", "json"]
        self.dump_format = dump_format
        self.sequence_data = {}  # For collecting JSON data per sequence
        if self.dump_results_path:
            os.makedirs(self.dump_results_path, exist_ok=True)

    def dump_results(self, result: dict, data_sample: dict):
        """Dump predictions to a pickle or JSON file."""
        img_path = data_sample["img_path"]
        parts = img_path.split("/")
        scene_id = parts[-2]
        frame_id = os.path.splitext(parts[-1])[0]

        # Apply score threshold filtering for dumping
        if self.score_thr > 0:
            valid_indices = np.where(result['scores'] > self.score_thr)[0]
            if len(valid_indices) > 0:
                filtered_result = {
                    "bboxes": result['bboxes'][valid_indices, :],
                    "scores": result['scores'][valid_indices],
                    "labels": result['labels'][valid_indices],
                    "translations": result['translations'][valid_indices, :],
                    "rotations": result['rotations'][valid_indices, :],
                    "sizes": result['sizes'][valid_indices, :],
                    "T": result['T'][valid_indices, :]
                }
            else:
                # No predictions above threshold, create empty arrays
                filtered_result = {
                    "bboxes": np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "labels": np.zeros(0, dtype=np.int32),
                    "translations": np.zeros((0, 3)),
                    "rotations": np.zeros((0, 3, 3)),
                    "sizes": np.zeros((0, 3)),
                    "T": np.zeros((0, 4, 4))
                }
        else:
            filtered_result = result

        if self.dump_format == "pickle":
            # Original pickle format
            dump_dir = os.path.join(self.dump_results_path, scene_id)
            os.makedirs(dump_dir, exist_ok=True)

            file_path = os.path.join(dump_dir, f"{frame_id}.pkl")

            dump_data = {
                "pred_class_ids": filtered_result["labels"],
                "pred_bboxes": filtered_result["bboxes"],
                "pred_scores": filtered_result["scores"],
                "pred_RTs": filtered_result["T"],
                "pred_scales": filtered_result["sizes"],
            }

            with open(file_path, "wb") as f:
                pickle.dump(dump_data, f)
                
        elif self.dump_format == "json":
            # JSON format - collect data for the sequence
            if scene_id not in self.sequence_data:
                self.sequence_data[scene_id] = {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {"id": i+1, "name": name} 
                        for i, name in enumerate(self.dataset_meta["classes"])
                    ]
                }
            
            # Get image info
            img_height, img_width = data_sample.get("img_shape", (480, 640))  # Default values
            
            # Extract camera intrinsics if available
            intrinsic = data_sample['intrinsic']
            if len(intrinsic) == 9:
                fx = intrinsic[0]
                fy = intrinsic[4]
                cx = intrinsic[2]
                cy = intrinsic[5]
            elif len(intrinsic) == 4:
                fx = intrinsic[0]
                fy = intrinsic[1]
                cx = intrinsic[2]
                cy = intrinsic[3]
       
            # Create image entry
            image_id = len(self.sequence_data[scene_id]["images"]) + 1
            image_entry = {
                "id": image_id,
                "file_name": f"{frame_id}.png",
                "width": img_width,
                "height": img_height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
            self.sequence_data[scene_id]["images"].append(image_entry)
            
            # Create annotation entries (only for filtered predictions)
            num_predictions = len(filtered_result["labels"])
            for i in range(num_predictions):
                annotation_id = len(self.sequence_data[scene_id]["annotations"]) + 1
                
                # Extract rotation matrix from transformation matrix
                T_matrix = filtered_result["T"][i]  # 4x4 transformation matrix
                translation = T_matrix[:3, 3].tolist()
                rotation_matrix = T_matrix[:3, :3]
                rot_scale = np.cbrt(np.linalg.det(rotation_matrix))
                rotation_matrix = (rotation_matrix / rot_scale).flatten().tolist()
                sizes = (filtered_result["sizes"][i] * rot_scale).tolist()
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(filtered_result["labels"][i]) + 1,  # Convert to 1-based indexing
                    "translation": translation,
                    "rotation_matrix": rotation_matrix,
                    "size": sizes,
                    "bbox": filtered_result["bboxes"][i].tolist(),
                    "score": float(filtered_result["scores"][i])  # Add prediction score
                }
                self.sequence_data[scene_id]["annotations"].append(annotation_entry)
                
    def finalize_json_dumps(self):
        """Write collected JSON data for all sequences."""
        if self.dump_format == "json" and self.sequence_data:
            for scene_id, data in self.sequence_data.items():
                json_path = os.path.join(self.dump_results_path, f"{scene_id}.json")
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
            print(f"JSON files saved for {len(self.sequence_data)} sequences in {self.dump_results_path}")

    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt["gt_instances"]
            gt_ignore_instances = gt["ignored_instances"]

            # ``model.test_step`` asks the head for predictions rescaled to
            # ``ori_shape``.  Keep GT in that same coordinate system.  This is
            # especially important for the 736x512 fruit data whose validation
            # pipeline resizes to 640x445 before packing annotations.
            metadata = getattr(data_sample, "metainfo", data_sample)
            scale_factor = metadata.get("scale_factor")

            ann = dict(
                labels=gt_instances["labels"].cpu().numpy(),
                bboxes=rescale_ground_truth_bboxes(
                    gt_instances["bboxes"].cpu().numpy(), scale_factor),
                bboxes_ignore=rescale_ground_truth_bboxes(
                    gt_ignore_instances["bboxes"].cpu().numpy(), scale_factor),
                labels_ignore=gt_ignore_instances["labels"].cpu().numpy(),
                translations=gt_instances["translations"].cpu().numpy(),
                rotations=gt_instances["rotations"].cpu().numpy(),
                sizes=gt_instances["sizes"].cpu().numpy(),
                T=gt_instances["T"].cpu().numpy(),
            )

            if "gt_handle_visibility" in gt_instances:
                ann["gt_handle_visibility"] = (
                    gt_instances["gt_handle_visibility"].cpu().numpy()
                )

            result = dict()
            pred = data_sample["pred_instances"]
            keep = select_aligned_prediction_indices(
                pred, score_thr=self.score_thr, nms_cfg=self.nms_cfg)
            for field in (
                "bboxes",
                "scores",
                "labels",
                "translations",
                "rotations",
                "sizes",
                "T",
            ):
                result[field] = pred[field][keep].cpu().numpy()

            # A diagnostic 2D AP sweep often needs all decoder queries, while
            # deployment/3D pose metrics should retain their score+NMS policy.
            # Keep only the three fields eval_map consumes in this optional
            # view; all 3D fields remain aligned to ``result`` above.
            if self.hbb_selection is not None:
                hbb_keep = select_aligned_prediction_indices(
                    pred,
                    score_thr=self.hbb_selection["score_thr"],
                    nms_cfg=self.hbb_selection["nms_cfg"],
                )
                result["hbb_eval"] = {
                    field: pred[field][hbb_keep].cpu().numpy()
                    for field in ("bboxes", "scores", "labels")
                }

            # gt_scale = np.linalg.norm(ann['sizes'], axis=1)
            pred_scale = np.linalg.norm(result["sizes"], axis=1)
            result["sizes"] = result["sizes"] / pred_scale[:, None]
            # ann['sizes'] = ann['sizes'] / gt_scale[:, None]
            # # gt_R = ann['T'][:, :3, :3] * gt_scale[:, None, None]
            pred_R = result["T"][:, :3, :3] * pred_scale[:, None, None]
            # # ann['T'][:, :3, :3] = gt_R
            result["T"][:, :3, :3] = pred_R

            if self.dump_results_path:
                self.dump_results(result, data_sample)

            self.results.append((ann, result))

    def compute_independent_mAP(
        self,
        preds,
        gts,
        degree_thresholds=[5, 10],
        shift_thresholds=[2, 5, 10],
        iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75],
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
        logger=None,
        cat_id=-1,
        classes=None,
        num_workers=None,
    ):
        if num_workers is None:
            num_workers = min(
                cpu_count(), 8
            )  # Limit to 8 processes to avoid memory issues

        total_images = len(preds)
        num_classes = len(classes)
        degree_thres_list = list(degree_thresholds) + [360]
        num_degree_thres = len(degree_thres_list)

        shift_thres_list = list(shift_thresholds) + [100]
        num_shift_thres = len(shift_thres_list)

        iou_thres_list = list(iou_3d_thresholds)
        num_iou_thres = len(iou_thres_list)

        if use_matches_for_pose:
            assert iou_pose_thres in iou_thres_list

        iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
        iou_pred_matches_all = [
            np.zeros((num_iou_thres, 0)) for _ in range(num_classes)
        ]
        iou_pred_scores_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
        iou_gt_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]

        pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
        pose_pred_matches_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]
        pose_gt_matches_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]
        pose_pred_scores_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]

        # Create smaller batches for more frequent progress updates
        # Use smaller batch size to get more granular progress feedback
        images_per_batch = max(
            1, min(10, total_images // (num_workers * 4))
        )  # 4x more batches than workers
        batches = [
            (preds[i : i + images_per_batch], gts[i : i + images_per_batch])
            for i in range(0, len(preds), images_per_batch)
        ]

        # Prepare worker function arguments
        worker_args = [
            (
                batch_preds,
                batch_gts,
                num_classes,
                classes,
                iou_thres_list,
                degree_thres_list,
                shift_thres_list,
                use_matches_for_pose,
                iou_pose_thres,
                self.two_phase_3d_iou,
            )
            for batch_preds, batch_gts in batches
        ]

        print(
            f"Processing {total_images} images across {len(batches)} batches ({images_per_batch} images per batch) with {num_workers} workers..."
        )

        # Process batches with image-level progress tracking
        if num_workers > 1 and len(batches) > 1:
            with Pool(num_workers) as pool:
                # Create a progress bar that tracks the total number of images
                with tqdm(
                    total=total_images, desc="Processing images", unit="img"
                ) as pbar:
                    batch_results = []
                    for i, batch_result in enumerate(
                        pool.imap(_process_batch_worker, worker_args)
                    ):
                        batch_results.append(batch_result)
                        # Update progress by the actual size of the current batch
                        current_batch_size = len(batches[i][0])  # batch_preds length
                        pbar.update(current_batch_size)
        else:
            # Fallback to sequential processing with image-level progress
            batch_results = []
            with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
                for i, args in enumerate(worker_args):
                    batch_result = _process_batch_worker(args)
                    batch_results.append(batch_result)
                    # Update progress by the actual size of the current batch
                    current_batch_size = len(batches[i][0])  # batch_preds length
                    pbar.update(current_batch_size)

        # Aggregate results from all batches
        print("Aggregating results from all batches...")
        pruning_stats = {"total_pairs": 0, "exact_candidates": 0}
        for batch_result in tqdm(
            batch_results, desc="Aggregating batches", unit="batch"
        ):
            (
                batch_iou_pred_matches,
                batch_iou_pred_scores,
                batch_iou_gt_matches,
                batch_pose_pred_matches,
                batch_pose_pred_scores,
                batch_pose_gt_matches,
                batch_pruning_stats,
            ) = batch_result
            pruning_stats["total_pairs"] += batch_pruning_stats["total_pairs"]
            pruning_stats["exact_candidates"] += batch_pruning_stats[
                "exact_candidates"
            ]

            for cls_id in range(num_classes):
                iou_pred_matches_all[cls_id] = np.concatenate(
                    (iou_pred_matches_all[cls_id], batch_iou_pred_matches[cls_id]),
                    axis=-1,
                )
                iou_pred_scores_all[cls_id] = np.concatenate(
                    (iou_pred_scores_all[cls_id], batch_iou_pred_scores[cls_id]),
                    axis=-1,
                )
                iou_gt_matches_all[cls_id] = np.concatenate(
                    (iou_gt_matches_all[cls_id], batch_iou_gt_matches[cls_id]), axis=-1
                )

                pose_pred_matches_all[cls_id] = np.concatenate(
                    (pose_pred_matches_all[cls_id], batch_pose_pred_matches[cls_id]),
                    axis=-1,
                )
                pose_pred_scores_all[cls_id] = np.concatenate(
                    (pose_pred_scores_all[cls_id], batch_pose_pred_scores[cls_id]),
                    axis=-1,
                )
                pose_gt_matches_all[cls_id] = np.concatenate(
                    (pose_gt_matches_all[cls_id], batch_pose_gt_matches[cls_id]),
                    axis=-1,
                )

        total_pairs = pruning_stats["total_pairs"]
        candidate_pairs = pruning_stats["exact_candidates"]
        pruning_message = (
            "3D IoU exact-phase candidates: "
            f"{candidate_pairs}/{total_pairs} "
            f"({candidate_pairs / total_pairs:.2%})"
            if total_pairs
            else "3D IoU exact-phase candidates: 0/0"
        )
        if logger is not None:
            logger.info(pruning_message)
        else:
            print(pruning_message)

        # Compute AP scores (this part remains sequential as it's already fast)
        print("Computing IoU AP scores...")
        iou_dict = {}
        iou_dict["thres_list"] = iou_thres_list
        for cls_id in tqdm(range(num_classes), desc="Computing IoU APs", unit="class"):
            for s, iou_thres in enumerate(iou_thres_list):
                iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(
                    iou_pred_matches_all[cls_id][s, :],
                    iou_pred_scores_all[cls_id][s, :],
                    iou_gt_matches_all[cls_id][s, :],
                )

        iou_3d_aps[-1, :] = np.mean(iou_3d_aps[:-1, :], axis=0)

        print("Computing pose AP scores...")
        total_pose_combinations = len(degree_thres_list) * len(shift_thres_list)
        with tqdm(
            total=total_pose_combinations, desc="Computing pose APs", unit="combination"
        ) as pbar:
            for i, degree_thres in enumerate(degree_thres_list):
                for j, shift_thres in enumerate(shift_thres_list):
                    for cls_id in range(num_classes):
                        cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][
                            i, j, :
                        ]
                        cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                        cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                        pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(
                            cls_pose_pred_matches_all,
                            cls_pose_pred_scores_all,
                            cls_pose_gt_matches_all,
                        )

                    pose_aps[-1, i, j] = np.mean(pose_aps[:-1, i, j])
                    pbar.update(1)

        if logger is not None:
            logger.warning(
                "3D IoU at 25: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.25)] * 100
                )
            )
            logger.warning(
                "3D IoU at 50: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.5)] * 100
                )
            )
            logger.warning(
                "3D IoU at 75: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.75)] * 100
                )
            )

            logger.warning(
                "5 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            logger.warning(
                "5 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            logger.warning(
                "10 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            logger.warning(
                "10 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            logger.warning(
                f"3D IoU at 25 per category:{iou_3d_aps[:, iou_thres_list.index(0.25)] * 100}"
            )
            logger.warning(
                f"3D IoU at 50 per category:{iou_3d_aps[:, iou_thres_list.index(0.5)] * 100}"
            )
            logger.warning(
                f"3D IoU at 75 per category:{iou_3d_aps[:, iou_thres_list.index(0.75)] * 100}"
            )

            logger.warning(
                f"5 degree, 2cm per category:{pose_aps[:, degree_thres_list.index(5), shift_thres_list.index(2)] * 100}"
            )
            logger.warning(
                f"5 degree, 5cm per category:{pose_aps[:, degree_thres_list.index(5), shift_thres_list.index(5)] * 100}"
            )
            logger.warning(
                f"10 degree, 2cm per category:{pose_aps[:, degree_thres_list.index(10), shift_thres_list.index(2)] * 100}"
            )
            logger.warning(
                f"10 degree, 5cm per category:{pose_aps[:, degree_thres_list.index(10), shift_thres_list.index(5)] * 100}"
            )

        else:
            print(
                "3D IoU at 25: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.25)] * 100
                )
            )
            print(
                "3D IoU at 50: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.5)] * 100
                )
            )
            print(
                "3D IoU at 75: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.75)] * 100
                )
            )

            print(
                "5 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            print(
                "5 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            print(
                "10 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            print(
                "10 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

        result = OrderedDict()
        for i, iou_thres in enumerate(iou_thres_list):
            result[f"3d_iou_{iou_thres:.2f}"] = iou_3d_aps[cat_id, i]

        for i, degree_thres in enumerate(degree_thres_list):
            for j, shift_thres in enumerate(shift_thres_list):
                result[f"pose {int(degree_thres)} degree, {int(shift_thres)}cm"] = (
                    pose_aps[cat_id, i, j]
                )

        return result

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if not results:
            raise ValueError("NOCSMetric requires at least one processed sample")
        gts, preds = zip(*results)
        eval_results = OrderedDict()

        # ``eval_map`` expects one (N, 5) bbox+score array per image/class.
        # This representation is independent of IoU, so construct it once and
        # reuse it for every configured threshold.
        class_preds = []
        for pred in preds:
            hbb_pred = pred.get("hbb_eval", pred)
            tmp_dets = []
            for label in range(len(self.dataset_meta["classes"])):
                index = np.where(hbb_pred["labels"] == label)[0]
                pred_bbox_scores = np.hstack(
                    [
                        hbb_pred["bboxes"][index],
                        hbb_pred["scores"][index].reshape((-1, 1)),
                    ]
                )
                tmp_dets.append(pred_bbox_scores)
            class_preds.append(tmp_dets)

        aps_by_iou = OrderedDict()
        for iou_thr in self.iou_thrs:
            logger.info(f"\n{'-' * 15}iou_thr: {iou_thr}{'-' * 15}")
            # Follow the official implementation,
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
            # we should use the legacy coordinate system in yopo 1.x,
            # which means w, h should be computed as 'x2 - x1 + 1` and
            # `y2 - y1 + 1`
            mean_ap, _ = eval_map(
                class_preds,
                gts,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=self.dataset_meta["classes"],
                logger=logger,
                eval_mode="area",
                use_legacy_coordinate=True,
            )
            # Keep full precision: this value drives checkpoint selection and
            # early stopping, where three-decimal rounding can turn genuine
            # small improvements into false plateaus. Logger formatting may
            # still present a compact value without changing the metric.
            mean_ap = float(mean_ap)
            if not np.isfinite(mean_ap):
                raise ValueError(
                    f"eval_map returned a non-finite AP at IoU {iou_thr}: "
                    f"{mean_ap}")
            aps_by_iou[iou_thr] = mean_ap
            eval_results[_ap_key(iou_thr)] = mean_ap

        # AP50_95 is the arithmetic mean of the ten dataset-level mAP values,
        # matching the aggregation convention used for COCO-style AP50:95.
        # It is emitted only for that exact range, so sparse/custom threshold
        # sets cannot accidentally masquerade as the canonical metric.
        if _is_coco_iou_range(self.iou_thrs):
            eval_results["AP50_95"] = float(np.mean(tuple(aps_by_iou.values())))
        logger.info(
            "3d_iou_* uses exact arbitrary-SO(3) oriented cuboid overlap; "
            "historical values from the pre-2026-08-26 corner-axis metric "
            "are not comparable."
        )
        pose_results = self.compute_independent_mAP(
            preds, gts, logger=logger, cat_id=-1, classes=self.dataset_meta["classes"]
        )
        eval_results.update(pose_results)

        # Finalize JSON dumps if using JSON format
        if self.dump_format == "json":
            self.finalize_json_dumps()

        return eval_results


def compute_3d_matches(
    gt_class_ids,
    gt_RTs,
    gt_scales,
    gt_handle_visibility,
    class_names,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_RTs,
    pred_scales,
    iou_3d_thresholds,
    score_threshold=0,
    prune_below_iou_threshold=False,
    return_pruning_stats=False,
):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps. When
            ``prune_below_iou_threshold`` is true, entries proven unable to
            reach the lowest requested threshold are zero without running the
            exact narrow phase. Matching/AP remains exact, but those
            below-threshold entries are not diagnostic exact IoUs.
        pruning_stats: Appended only when ``return_pruning_stats=True``.
    """

    def trim_zeros(x):
        """It's common to have tensors larger than the available data and
        pad with zeros. This function removes rows that are all zeros.
        x: [rows, columns].
        """

        pre_shape = x.shape
        assert len(x.shape) == 2, x.shape
        new_x = x[~np.all(x == 0, axis=1)]
        post_shape = new_x.shape
        assert pre_shape[0] == post_shape[0]
        assert pre_shape[1] == post_shape[1]

        return new_x

    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[: pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Decompose each similarity transform once.  Convex overlap remains
    # pairwise, but repeating SVD/polar normalization for every N x M pair is
    # unnecessarily expensive on dense fruit scenes.
    pred_oriented_boxes = [
        None
        if pred_RTs[i] is None
        else _decompose_nocs_similarity_box(
            pred_RTs[i],
            pred_scales[i],
            name=f"pred_RTs[{i}]",
            allow_nonpositive_sizes=True,
        )
        for i in range(num_pred)
    ]
    gt_oriented_boxes = [
        None
        if gt_RTs[j] is None
        else _decompose_nocs_similarity_box(
            gt_RTs[j], gt_scales[j], name=f"gt_RTs[{j}]"
        )
        for j in range(num_gt)
    ]

    if not iou_3d_thresholds:
        raise ValueError("iou_3d_thresholds must contain at least one value")
    lowest_iou_threshold = float(min(iou_3d_thresholds))
    if not np.isfinite(lowest_iou_threshold) or lowest_iou_threshold < 0.0:
        raise ValueError(
            "iou_3d_thresholds must contain finite non-negative values"
        )
    if prune_below_iou_threshold and lowest_iou_threshold > 0.0:
        exact_candidate_mask, _ = _pairwise_exact_iou_candidate_mask(
            pred_oriented_boxes,
            gt_oriented_boxes,
            lowest_iou_threshold=lowest_iou_threshold,
            pred_class_ids=pred_class_ids,
            gt_class_ids=gt_class_ids,
            class_names=class_names,
            gt_handle_visibility=gt_handle_visibility,
        )
    else:
        exact_candidate_mask = np.ones((num_pred, num_gt), dtype=bool)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            if exact_candidate_mask[i, j]:
                overlaps[i, j] = _compute_decomposed_3d_iou(
                    pred_oriented_boxes[i],
                    gt_oriented_boxes[j],
                    gt_handle_visibility[j],
                    class_names[pred_class_ids[i]],
                    class_names[gt_class_ids[j]],
                )
            elif pred_oriented_boxes[i] is None or gt_oriented_boxes[j] is None:
                overlaps[i, j] = -1.0

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    sorted_overlap_indices = []
    for i in range(len(pred_boxes)):
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[: low_score_idx[0]]
        sorted_overlap_indices.append(sorted_ixs)

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find the match using the threshold-independent cached ordering.
            for j in sorted_overlap_indices[i]:
                # If ground truth box is already matched, go to next one
                # print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    result = (gt_matches, pred_matches, overlaps, indices)
    if return_pruning_stats:
        result += (
            {
                "total_pairs": int(num_pred * num_gt),
                "exact_candidates": int(np.count_nonzero(exact_candidate_mask)),
            },
        )
    return result


def compute_RT_overlaps(
    gt_class_ids, gt_RTs, gt_handle_visibility, pred_class_ids, pred_RTs, class_names
):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(
                pred_RTs[i],
                gt_RTs[j],
                gt_class_ids[j],
                gt_handle_visibility[j],
                class_names,
            )
    return overlaps


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, class_names):
    """
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    class_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    class_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    """

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if class_names[class_id] in ["bottle", "can", "bowl"]:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif class_names[class_id] == "mug" and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_names[class_id] in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(
            np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2)
        )
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])
    return result


def compute_match_from_degree_cm(
    overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list
):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    # print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if (
                        overlaps[i, j, 0] > degree_thres
                        or overlaps[i, j, 1] > shift_thres
                    ):
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def _decompose_nocs_similarity_box(
    RT, scales, *, name, allow_nonpositive_sizes=False
):
    """Return ``(center, full_sizes, SO(3) rotation)`` from a NOCS RT."""
    transform = np.asarray(RT, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} must contain only finite values")

    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError(
            f"{name} linear block must have non-zero finite determinant, "
            f"got {determinant}"
        )
    uniform_scale = float(np.cbrt(determinant))
    rotation = linear / uniform_scale
    gram = rotation.T @ rotation
    # Float32 6D-to-SO(3) conversion reaches ~2e-5 Gram error on the current
    # validation predictions.  A 1e-4 gate admits that measured numerical
    # drift while still rejecting genuinely anisotropic affine transforms.
    if not np.allclose(gram, np.eye(3), rtol=1e-4, atol=1e-4):
        raise ValueError(
            f"{name} linear block must be rotation times uniform scale; "
            f"max|R^T R-I|={np.max(np.abs(gram - np.eye(3))):.12g}"
        )

    # Project only the already-validated near-rotation to exact SO(3), so the
    # geometry helper can retain its strict input contract.
    left, _, right_t = np.linalg.svd(rotation)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t

    full_sizes = np.asarray(scales, dtype=np.float64)
    if full_sizes.shape != (3,):
        raise ValueError(f"{name} scales must have shape (3,), got {full_sizes.shape}")
    full_sizes = abs(uniform_scale) * full_sizes
    if not np.isfinite(full_sizes).all():
        raise ValueError(
            f"{name} effective full sizes must be finite, "
            f"got {full_sizes.tolist()}"
        )
    if np.any(full_sizes <= 0.0):
        if allow_nonpositive_sizes:
            return transform[:3, 3], None, rotation
        raise ValueError(
            f"{name} effective full sizes must be positive, "
            f"got {full_sizes.tolist()}"
        )
    return transform[:3, 3], full_sizes, rotation


def _uses_nocs_continuous_y_symmetry(
    class_name_1, class_name_2, handle_visibility
):
    return (
        class_name_1 in ["bottle", "bowl", "can"]
        and class_name_1 == class_name_2
    ) or (
        class_name_1 == "mug"
        and class_name_1 == class_name_2
        and handle_visibility == 0
    )


def _pairwise_exact_iou_candidate_mask(
    pred_boxes,
    gt_boxes,
    *,
    lowest_iou_threshold,
    pred_class_ids,
    gt_class_ids,
    class_names,
    gt_handle_visibility,
):
    """Return safe exact-phase candidates and their IoU upper bounds."""
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    upper_bounds = np.full((num_pred, num_gt), -1.0, dtype=np.float64)
    valid_pred = [
        index
        for index, box in enumerate(pred_boxes)
        if box is not None and box[1] is not None
    ]
    valid_gt = [index for index, box in enumerate(gt_boxes) if box is not None]
    if valid_pred and valid_gt:
        pred_centers = np.stack([pred_boxes[index][0] for index in valid_pred])
        pred_sizes = np.stack([pred_boxes[index][1] for index in valid_pred])
        pred_rotations = np.stack([pred_boxes[index][2] for index in valid_pred])
        gt_centers = np.stack([gt_boxes[index][0] for index in valid_gt])
        gt_sizes = np.stack([gt_boxes[index][1] for index in valid_gt])
        gt_rotations = np.stack([gt_boxes[index][2] for index in valid_gt])
        valid_upper = pairwise_oriented_box_iou_upper_bound_from_boxes_3d(
            pred_centers,
            pred_sizes,
            pred_rotations,
            gt_centers,
            gt_sizes,
            gt_rotations,
        )
        upper_bounds[np.ix_(valid_pred, valid_gt)] = valid_upper

    # A pair-specific local-y canonicalization can change an anisotropic box's
    # world AABB.  Do not prune declared symmetric pairs using the unaligned
    # envelope; send them directly to the exact symmetry-aware narrow phase.
    symmetry_pairs = np.zeros((num_pred, num_gt), dtype=bool)
    for pred_index in valid_pred:
        for gt_index in valid_gt:
            symmetry_pairs[pred_index, gt_index] = (
                _uses_nocs_continuous_y_symmetry(
                    class_names[pred_class_ids[pred_index]],
                    class_names[gt_class_ids[gt_index]],
                    gt_handle_visibility[gt_index],
                )
            )
    upper_bounds[symmetry_pairs] = 1.0

    # Strict '<' pruning retains equality-boundary pairs, including any one-ULP
    # numerical uncertainty introduced while constructing the AABB envelope.
    candidate_mask = upper_bounds >= lowest_iou_threshold
    return candidate_mask, upper_bounds


def _compute_decomposed_3d_iou(
    box_1, box_2, handle_visibility, class_name_1, class_name_2
):
    """Compute IoU from validated ``(center, size, rotation)`` tuples."""
    if box_1 is None or box_2 is None:
        return -1

    center_1, full_sizes_1, rotation_1 = box_1
    center_2, full_sizes_2, rotation_2 = box_2
    if full_sizes_1 is None:
        # Direct size regression can occasionally emit a finite non-positive
        # side.  Keep it as an unmatched false positive instead of aborting a
        # whole validation run.  GT decomposition remains strict above.
        return 0.0

    if _uses_nocs_continuous_y_symmetry(
        class_name_1, class_name_2, handle_visibility
    ):
        # Preserve the established NOCS continuous local-y symmetry policy.
        relative_rotation = rotation_1.T @ rotation_2
        theta = np.arctan2(
            relative_rotation[0, 2] - relative_rotation[2, 0],
            relative_rotation[0, 0] + relative_rotation[2, 2],
        )
        cosine = np.cos(theta)
        sine = np.sin(theta)
        rotation_1 = rotation_1 @ np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )

    return oriented_box_iou_3d(
        center_1,
        full_sizes_1,
        rotation_1,
        center_2,
        full_sizes_2,
        rotation_2,
    )


def compute_3d_iou(
    RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2
):
    """Compute exact IoU between two NOCS similarity-transform cuboids.

    The linear block of each ``RT`` is interpreted as a proper rotation times
    one uniform scalar.  That scalar is moved into the supplied full side
    lengths before evaluating the true arbitrary-SO(3) cuboid intersection.
    A small polar projection removes floating-point drift only after the
    similarity contract has been validated.

    ``3d_iou_*`` values emitted through this function are intentionally not
    comparable with historical runs that reduced transformed corners along
    the wrong array axis.  ``None`` transforms retain the historical ``-1``
    sentinel.  A finite non-positive predicted side produces IoU zero, leaving
    that prediction to be counted as a false positive; invalid GT sizes,
    non-finite values, and malformed/non-similarity transforms fail fast.
    """

    if RT_1 is None or RT_2 is None:
        return -1

    center_1, full_sizes_1, rotation_1 = _decompose_nocs_similarity_box(
        RT_1,
        scales_1,
        name="RT_1",
        allow_nonpositive_sizes=True,
    )
    center_2, full_sizes_2, rotation_2 = _decompose_nocs_similarity_box(
        RT_2, scales_2, name="RT_2"
    )
    return _compute_decomposed_3d_iou(
        (center_1, full_sizes_1, rotation_1),
        (center_2, full_sizes_2, rotation_2),
        handle_visibility,
        class_name_1,
        class_name_2,
    )


def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = (
            np.array(
                [
                    [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                ]
            )
            + shift
        )
    else:
        bbox_3d = (
            np.array(
                [
                    [scale / 2, +scale / 2, scale / 2],
                    [scale / 2, +scale / 2, -scale / 2],
                    [-scale / 2, +scale / 2, scale / 2],
                    [-scale / 2, +scale / 2, -scale / 2],
                    [+scale / 2, -scale / 2, scale / 2],
                    [+scale / 2, -scale / 2, -scale / 2],
                    [-scale / 2, -scale / 2, scale / 2],
                    [-scale / 2, -scale / 2, -scale / 2],
                ]
            )
            + shift
        )

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def _process_batch_worker(args):
    """Worker function for parallel processing of prediction batches."""
    (
        batch_preds,
        batch_gts,
        num_classes,
        classes,
        iou_thres_list,
        degree_thres_list,
        shift_thres_list,
        use_matches_for_pose,
        iou_pose_thres,
        two_phase_3d_iou,
    ) = args

    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_iou_thres = len(iou_thres_list)

    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]

    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pose_pred_scores_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pruning_stats = {"total_pairs": 0, "exact_candidates": 0}

    for pred, gt in zip(batch_preds, batch_gts):
        gt_class_ids = gt["labels"].astype(np.int32)
        gt_RTs = gt["T"]
        gt_sizes = gt["sizes"]

        if "gt_handle_visibility" in gt:
            gt_handle_visibility = gt["gt_handle_visibility"]
        else:
            gt_handle_visibility = np.ones_like(gt_class_ids)

        pred_bboxes = np.array(pred["bboxes"])
        pred_class_ids = pred["labels"]
        pred_scores = pred["scores"]
        pred_RTs = pred["T"]
        pred_sizes = pred["sizes"]

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = (
                gt_class_ids[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros(0)
            )
            cls_gt_scales = (
                gt_sizes[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 3))
            )
            cls_gt_RTs = (
                gt_RTs[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 4, 4))
            )

            cls_pred_class_ids = (
                pred_class_ids[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros(0)
            )
            cls_pred_bboxes = (
                pred_bboxes[pred_class_ids == cls_id, :]
                if len(pred_class_ids)
                else np.zeros((0, 4))
            )
            cls_pred_scores = (
                pred_scores[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros(0)
            )
            cls_pred_RTs = (
                pred_RTs[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 4, 4))
            )
            cls_pred_scales = (
                pred_sizes[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 3))
            )

            # calculate the overlap between each gt instance and pred instance
            if classes[cls_id] != "mug":
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = (
                    gt_handle_visibility[gt_class_ids == cls_id]
                    if len(gt_class_ids)
                    else np.ones(0)
                )

            (
                iou_cls_gt_match,
                iou_cls_pred_match,
                _,
                iou_pred_indices,
                class_pruning_stats,
            ) = (
                compute_3d_matches(
                    cls_gt_class_ids,
                    cls_gt_RTs,
                    cls_gt_scales,
                    cls_gt_handle_visibility,
                    classes,
                    cls_pred_bboxes,
                    cls_pred_class_ids,
                    cls_pred_scores,
                    cls_pred_RTs,
                    cls_pred_scales,
                    iou_thres_list,
                    prune_below_iou_threshold=two_phase_3d_iou,
                    return_pruning_stats=True,
                )
            )
            pruning_stats["total_pairs"] += class_pruning_stats["total_pairs"]
            pruning_stats["exact_candidates"] += class_pruning_stats[
                "exact_candidates"
            ]
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]

            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1
            )
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate(
                (iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1
            )
            assert (
                iou_pred_matches_all[cls_id].shape[1]
                == iou_pred_scores_all[cls_id].shape[1]
            )
            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1
            )

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                cls_pred_class_ids = (
                    cls_pred_class_ids[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros(0)
                )
                cls_pred_RTs = (
                    cls_pred_RTs[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4, 4))
                )
                cls_pred_scores = (
                    cls_pred_scores[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros(0)
                )
                cls_pred_bboxes = (
                    cls_pred_bboxes[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4))
                )

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = (
                    cls_gt_class_ids[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros(0)
                )
                cls_gt_RTs = (
                    cls_gt_RTs[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros((0, 4, 4))
                )
                cls_gt_handle_visibility = (
                    cls_gt_handle_visibility[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros(0)
                )

            RT_overlaps = compute_RT_overlaps(
                cls_gt_class_ids,
                cls_gt_RTs,
                cls_gt_handle_visibility,
                cls_pred_class_ids,
                cls_pred_RTs,
                classes,
            )

            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(
                RT_overlaps,
                cls_pred_class_ids,
                cls_gt_class_ids,
                degree_thres_list,
                shift_thres_list,
            )

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1
            )

            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, 1)
            )
            pose_pred_scores_all[cls_id] = np.concatenate(
                (pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1
            )
            assert (
                pose_pred_scores_all[cls_id].shape[2]
                == pose_pred_matches_all[cls_id].shape[2]
            ), "{} vs. {}".format(
                pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape
            )
            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1
            )

    return (
        iou_pred_matches_all,
        iou_pred_scores_all,
        iou_gt_matches_all,
        pose_pred_matches_all,
        pose_pred_scores_all,
        pose_gt_matches_all,
        pruning_stats,
    )
