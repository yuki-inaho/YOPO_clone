"""Rotated AP for the GauCho ellipse head, scored against OBB annotations.

The 2D reference implementation reports rotated mAP50 between the *oriented
envelope of the learned ellipse* and the ground-truth OBB.  This metric
reproduces that definition inside YOPO so the two numbers are comparable on a
common split.

Both sides go through the same decode: a symmetric 2x2 shape matrix becomes
``(a, b, cx, cy, theta)`` and then the circumscribing ``(cx, cy, 2a, 2b,
theta)``.  Nothing here regresses or fits a box -- the envelope is a
deterministic function of the ellipse, exactly as on the reference side.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from mmcv.ops import box_iou_rotated, nms_rotated

from yopo.registry import METRICS

from .rotated_iou_metric import RotatedIoUMetric, _as_cpu_tensor, _field

__all__ = [
    "EllipseEnvelopeRotatedIoUMetric",
    "compact_gaussian_to_obb",
    "ellipse_error_decomposition",
    "rescale_compact_gaussian",
]


def rescale_compact_gaussian(compact: torch.Tensor,
                             scale_factor) -> torch.Tensor:
    """Map a compact Gaussian from transformed pixels back to original pixels.

    ``predict(rescale=True)`` returns detections in the *original* image frame,
    while ``gt_instances`` carries whatever frame the val pipeline produced --
    here 640x445 after ``ResizeforPose``, not the native 736x512.  Comparing the
    two directly costs roughly a 1.15x inflation, which shows up as a handful of
    near-centre matches and almost no recall.

    The centre maps as ``S mu`` and the shape as ``S Sigma S^T`` with
    ``S = diag(1/sx, 1/sy)``.  Doing it on the Gaussian rather than on decoded
    ``(w, h, theta)`` stays exact when the two axis scales differ.
    """
    if compact.numel() == 0:
        return compact
    sx, sy = float(scale_factor[0]), float(scale_factor[1])
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return compact
    inv_x, inv_y = 1.0 / sx, 1.0 / sy
    return torch.stack(
        (
            compact[:, 0] * inv_x,
            compact[:, 1] * inv_y,
            compact[:, 2] * inv_x * inv_x,
            compact[:, 3] * inv_x * inv_y,
            compact[:, 4] * inv_y * inv_y,
        ),
        dim=-1,
    )


def compact_gaussian_to_obb(compact: torch.Tensor) -> torch.Tensor:
    """``(cx, cy, xx, xy, yy)`` to the oriented envelope ``(cx, cy, w, h, t)``.

    ``Sigma`` of an OBB annotation is ``R diag((w/2)^2, (h/2)^2) R^T``, so its
    eigenvalues recover the semi-axes and its eigenvectors the orientation.
    The closed form below avoids an eigensolver and stays stable for the
    near-circular case, where the angle is simply not identified.
    """
    if compact.ndim != 2 or compact.shape[-1] != 5:
        raise ValueError(
            f"compact Gaussians must have shape (N, 5), got "
            f"{tuple(compact.shape)}")
    cx, cy = compact[:, 0], compact[:, 1]
    xx, xy, yy = compact[:, 2], compact[:, 3], compact[:, 4]
    trace = xx + yy
    discriminant = ((xx - yy).square() + 4.0 * xy.square()).clamp_min(0.0).sqrt()
    major = ((trace + discriminant) * 0.5).clamp_min(0.0).sqrt()
    minor = ((trace - discriminant) * 0.5).clamp_min(0.0).sqrt()
    theta = 0.5 * torch.atan2(2.0 * xy, xx - yy)
    return torch.stack((cx, cy, 2.0 * major, 2.0 * minor, theta), dim=-1)



def ellipse_error_decomposition(predicted: torch.Tensor,
                                target: torch.Tensor) -> dict:
    """Split the gap between two matched rotated boxes into its causes.

    A single mean IoU says a prediction is loose without saying how.  Raising
    the ellipse loss weight threefold moved the matched IoU by 0.0006, which
    means the residual is structural rather than a matter of optimisation
    pressure -- so it has to be decomposed before anything else is tried.

    Both inputs are ``(cx, cy, w, h, theta)`` with ``w >= h`` by construction of
    :func:`compact_gaussian_to_obb`.  Angles are compared modulo pi, because an
    extent has no front and a box rotated by pi is the same box.
    """
    centre_error = (predicted[:, :2] - target[:, :2]).norm(dim=-1)
    target_scale = (target[:, 2] * target[:, 3]).clamp_min(1e-9).sqrt()
    angle = (predicted[:, 4] - target[:, 4]).remainder(torch.pi)
    angle = torch.minimum(angle, torch.pi - angle)
    return {
        "centre_error_px": centre_error,
        "centre_error_rel": centre_error / target_scale,
        "long_ratio": predicted[:, 2] / target[:, 2].clamp_min(1e-9),
        "short_ratio": predicted[:, 3] / target[:, 3].clamp_min(1e-9),
        "area_ratio": ((predicted[:, 2] * predicted[:, 3])
                       / (target[:, 2] * target[:, 3]).clamp_min(1e-9)),
        "aspect_ratio_pred": predicted[:, 2] / predicted[:, 3].clamp_min(1e-9),
        "aspect_ratio_gt": target[:, 2] / target[:, 3].clamp_min(1e-9),
        "angle_error_deg": angle * (180.0 / torch.pi),
    }

@METRICS.register_module()
class EllipseEnvelopeRotatedIoUMetric(RotatedIoUMetric):
    """Rotated AP between predicted ellipse envelopes and annotated OBBs.

    Prediction field: ``pred_instances.ellipse_obb`` (already an envelope, from
    the head's GauCho decode).  Annotation field:
    ``gt_instances.obb_gaussians``, the compact Gaussian of the labelled OBB.

    A frame whose annotation is missing the OBB Gaussian is an error rather
    than an empty ground truth: silently scoring against nothing would inflate
    precision.


    ``nms_iou_threshold`` is off by default so the reported number stays the raw
    one every existing run logged.  Turn it on to compare against a detector
    that suppresses: the reference this work is measured against evaluates at
    rotated NMS IoU 0.2, and :class:`NOCSMetric` in this same repository already
    defaults to ``score_thr=0.2`` with ``nms_cfg=dict(type="nms",
    iou_threshold=0.5)``.  Scoring an unsuppressed 256-query emission against
    those is not a like-for-like comparison.

    ``geometry_oracle`` answers a different question from the rest of this
    class: not how good the geometry is, but how much of the remaining gap
    it could possibly close.  Every prediction that overlaps an annotation
    at all has its box replaced by that annotation's box, while its score,
    its class and its existence as a duplicate are left alone.  The
    resulting AP is what this detector would score with *perfect* ellipse
    geometry, so the distance from there to the target is the part geometry
    cannot reach -- detection, ranking and duplication.  A recall at a loose
    IoU threshold is not the same thing and does not bound the achievable AP.
    """

    def __init__(self,
                 pred_field: str = "ellipse_obb",
                 gt_field: str = "obb_gaussians",
                 nms_iou_threshold: Optional[float] = None,
                 geometry_oracle: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.geometry_oracle = bool(geometry_oracle)
        if nms_iou_threshold is not None and not 0.0 < nms_iou_threshold <= 1.0:
            raise ValueError(
                "nms_iou_threshold must be in (0, 1] or None, got "
                f"{nms_iou_threshold}")
        self.pred_field = pred_field
        self.gt_field = gt_field
        self.nms_iou_threshold = nms_iou_threshold

    def _suppress(self, boxes: torch.Tensor, scores: torch.Tensor,
                  labels: torch.Tensor):
        """Class-aware rotated NMS, or a no-op when the threshold is unset.

        ``mmcv.ops.nms_rotated`` accepts a ``labels`` argument but ignores it in
        this version -- two identical boxes with different labels still suppress
        each other.  Verified, so class separation is done here by shifting each
        class into its own region of the plane, the same trick
        ``batched_nms`` uses: boxes of different classes then cannot overlap and
        cannot suppress one another.
        """
        if self.nms_iou_threshold is None or boxes.numel() == 0:
            return boxes, scores, labels
        boxes = boxes.float()
        scores = scores.float()
        if self.num_classes > 1:
            # A stride wider than any box the image can hold.
            span = boxes[:, :2].abs().max() + boxes[:, 2:4].abs().max()
            offset = labels.to(boxes.dtype).unsqueeze(-1) * (span * 2.0 + 1.0)
            shifted = boxes.clone()
            shifted[:, :2] = shifted[:, :2] + offset
        else:
            shifted = boxes
        _, keep = nms_rotated(shifted, scores, self.nms_iou_threshold)
        return boxes[keep], scores[keep], labels[keep]

    def process(self, data_batch: dict, data_samples: Sequence[Any]) -> None:
        for data_sample in data_samples:
            pred = data_sample["pred_instances"]
            gt = data_sample["gt_instances"]
            try:
                pred_boxes = _as_cpu_tensor(
                    _field(pred, self.pred_field), torch.float32)
            except (KeyError, AttributeError) as error:
                raise KeyError(
                    f"pred_instances.{self.pred_field} is missing; enable "
                    "expose_gaucho_predictions on the head") from error
            try:
                gt_compact = _as_cpu_tensor(
                    _field(gt, self.gt_field), torch.float32)
            except (KeyError, AttributeError) as error:
                raise KeyError(
                    f"gt_instances.{self.gt_field} is missing; load the "
                    "annotation with with_obb_gaussian=True") from error

            if pred_boxes.numel() == 0:
                pred_boxes = pred_boxes.reshape(0, 5)
            if gt_compact.numel() == 0:
                gt_boxes = gt_compact.reshape(0, 5)
            else:
                scale_factor = (
                    data_sample.get("scale_factor")
                    if hasattr(data_sample, "get") else None)
                if scale_factor is not None:
                    gt_compact = rescale_compact_gaussian(
                        gt_compact, scale_factor)
                gt_boxes = compact_gaussian_to_obb(gt_compact)

            pred_boxes, pred_scores, pred_labels = self._suppress(
                pred_boxes,
                _as_cpu_tensor(_field(pred, "scores"), torch.float32),
                _as_cpu_tensor(_field(pred, "labels"), torch.long),
            )

            if self.geometry_oracle and len(pred_boxes) and len(gt_boxes):
                gt_labels_cpu = _as_cpu_tensor(
                    _field(gt, "labels"), torch.long)
                ious = box_iou_rotated(pred_boxes.float(),
                                       gt_boxes.float(), mode="iou",
                                       aligned=False, clockwise=True)
                same_class = (pred_labels[:, None]
                              == gt_labels_cpu[None, :])
                ious = torch.where(same_class, ious,
                                   torch.zeros_like(ious))
                best_iou, best_gt = ious.max(dim=1)
                # Only a prediction that already touches its object is
                # snapped; one that overlaps nothing is a false positive no
                # geometry can rescue, and leaving it keeps that honest.
                touches = best_iou > 0
                pred_boxes = pred_boxes.clone()
                pred_boxes[touches] = gt_boxes[best_gt[touches]]

            self.results.append(
                dict(
                    pred_bboxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_bboxes=gt_boxes,
                    gt_labels=_as_cpu_tensor(
                        _field(gt, "labels"), torch.long),
                    ignore_bboxes=torch.empty((0, 5)),
                    ignore_labels=torch.empty((0,), dtype=torch.long),
                ))

    def compute_metrics(self, results: list) -> dict:
        """Ranked AP, plus a breakdown of where the residual overlap goes.

        A mean matched IoU of 0.70 says predictions are loose without saying
        how, and tripling the ellipse loss weight moved it by 0.0006 -- so the
        residual is structural, and the only way forward is to see whether it
        sits in the centre, the extent, or the orientation.  The matching here
        is the same greedy, score-ordered, one-GT-per-prediction rule the AP
        above uses, so the decomposition describes exactly the pairs that were
        counted as true positives.
        """
        metrics = super().compute_metrics(results)

        decomposed: dict[str, list[torch.Tensor]] = {}
        for result in results:
            predicted = result['pred_bboxes']
            target = result['gt_bboxes']
            if predicted.numel() == 0 or target.numel() == 0:
                continue
            keep = result['pred_scores'] >= self.score_thr
            if not bool(keep.any()):
                continue
            predicted = predicted[keep]
            scores = result['pred_scores'][keep]
            ious = box_iou_rotated(predicted.float(), target.float(),
                                   mode='iou', aligned=False, clockwise=True)
            taken = torch.zeros(len(target), dtype=torch.bool)
            pairs: list[tuple[int, int]] = []
            for index in torch.argsort(scores, descending=True).tolist():
                candidates = torch.where(taken, torch.full_like(ious[index], -1.0),
                                         ious[index])
                best = int(candidates.argmax())
                if float(candidates[best]) >= self.iou_thr:
                    taken[best] = True
                    pairs.append((index, best))
            if not pairs:
                continue
            pred_index = torch.tensor([a for a, _ in pairs])
            gt_index = torch.tensor([b for _, b in pairs])
            for key, value in ellipse_error_decomposition(
                    predicted[pred_index], target[gt_index]).items():
                decomposed.setdefault(key, []).append(value)

        for key, chunks in decomposed.items():
            values = torch.cat(chunks)
            metrics[f'{key}_median'] = float(values.median())
            metrics[f'{key}_p90'] = float(values.quantile(0.9))

        # Is the orientation error reducible at all?  For a near-circular
        # annotation the OBB's orientation is barely identified -- the
        # annotator's choice is close to arbitrary -- so an angle error there
        # is a property of the data, not a deficiency of the model.  Splitting
        # the angle error by the *ground truth* aspect ratio separates the two:
        # if elongated objects are already accurate and only round ones are
        # wrong, there is nothing left to learn.
        if 'angle_error_deg' in decomposed and 'aspect_ratio_gt' in decomposed:
            angle = torch.cat(decomposed['angle_error_deg'])
            aspect = torch.cat(decomposed['aspect_ratio_gt'])
            for low, high, name in ((1.0, 1.1, 'round'),
                                    (1.1, 1.3, 'mid'),
                                    (1.3, 1.6, 'oval'),
                                    (1.6, float('inf'), 'elongated')):
                band = (aspect >= low) & (aspect < high)
                count = int(band.sum())
                metrics[f'angle_by_aspect_{name}_count'] = float(count)
                if count:
                    metrics[f'angle_by_aspect_{name}_median'] = float(
                        angle[band].median())
        return metrics
