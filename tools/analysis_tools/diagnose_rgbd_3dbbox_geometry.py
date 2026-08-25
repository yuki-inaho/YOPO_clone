#!/usr/bin/env python3
"""Audit RGB-D 3D-box geometry, native matching costs, and loss scales."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

from tools.analysis_tools.render_rgbd_3dbbox_overlays import (
    EDGES,
    cuboid_corners,
    intrinsic_matrix,
    project_corners,
)
from yopo.utils import (register_all_modules,
                        register_mmengine_checkpoint_safe_globals)


LOSS_NAMES = (
    "loss_cls",
    "loss_bbox",
    "loss_iou",
    "loss_centers_2d",
    "loss_z",
    "loss_rotation",
    "loss_size",
    "loss_size_chain",
    "loss_rotation_chain",
    "loss_z_chain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="resolved RGB-D model config")
    parser.add_argument("checkpoint", help="checkpoint to inspect")
    parser.add_argument("output_dir", help="report and overlay destination")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--overlay-images", type=int, default=5)
    return parser.parse_args()


def summarize_values(values: np.ndarray | Iterable[float]) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    summary = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.size:
        summary.update(
            min=float(finite.min()),
            median=float(np.median(finite)),
            max=float(finite.max()),
            mean=float(finite.mean()),
            std=float(finite.std()),
        )
    return summary


def evaluate_geometry_contract(
    *,
    intrinsic: object,
    translations: np.ndarray,
    transforms: np.ndarray,
    centers_2d: np.ndarray,
    log_z: np.ndarray,
    sizes: np.ndarray,
    z_is_log: bool = True,
) -> dict:
    """Validate camera-frame metres, log-depth, center, and cuboid projection."""
    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    transforms = np.asarray(transforms, dtype=np.float64).reshape(-1, 4, 4)
    centers_2d = np.asarray(centers_2d, dtype=np.float64).reshape(-1, 2)
    log_z = np.asarray(log_z, dtype=np.float64).reshape(-1, 1)
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1, 3)
    count = len(translations)
    if not (
        len(transforms) == len(centers_2d) == len(log_z) == len(sizes) == count
    ):
        raise ValueError("GT geometry arrays must have identical instance counts")

    k = intrinsic_matrix(intrinsic)
    arrays = (translations, transforms, centers_2d, log_z, sizes, k)
    all_finite = all(np.isfinite(value).all() for value in arrays)
    positive_translation_depth = bool(
        count == 0 or np.all(translations[:, 2] > 1e-6)
    )
    positive_sizes = bool(count == 0 or np.all(sizes > 0.0))

    t_error = (
        np.linalg.norm(transforms[:, :3, 3] - translations, axis=1)
        if count
        else np.empty(0)
    )
    if count and positive_translation_depth:
        expected_z = (
            np.log(translations[:, 2:3])
            if z_is_log
            else translations[:, 2:3]
        )
        log_z_error = np.abs(expected_z - log_z).reshape(-1)
        projected = (k @ translations.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        center_error = np.linalg.norm(projected - centers_2d, axis=1)
    else:
        log_z_error = np.full(count, np.inf)
        center_error = np.full(count, np.inf)

    positive_cuboid_depth = True
    projection_finite = True
    for size, transform in zip(sizes, transforms):
        try:
            corners = cuboid_corners(size, transform)
            positive_cuboid_depth &= bool(np.all(corners[:, 2] > 1e-6))
            projected = project_corners(corners, k)
            projection_finite &= bool(np.isfinite(projected).all())
        except (ValueError, FloatingPointError):
            positive_cuboid_depth = False
            projection_finite = False

    max_t_error = float(t_error.max()) if t_error.size else 0.0
    max_log_z_error = float(log_z_error.max()) if log_z_error.size else 0.0
    max_center_error = float(center_error.max()) if center_error.size else 0.0
    passed = bool(
        all_finite
        and positive_translation_depth
        and positive_sizes
        and positive_cuboid_depth
        and projection_finite
        and max_t_error <= 1e-6
        and max_log_z_error <= 1e-6
        and max_center_error <= 1.0
    )
    return {
        "pass": passed,
        "instance_count": count,
        "all_finite": all_finite,
        "positive_translation_depth": positive_translation_depth,
        "positive_sizes": positive_sizes,
        "positive_cuboid_depth": positive_cuboid_depth,
        "projection_finite": projection_finite,
        "max_translation_T_error_m": max_t_error,
        "max_log_z_error": max_log_z_error,
        "z_encoding": "log_metres" if z_is_log else "metres",
        "max_center_projection_error_px": max_center_error,
    }


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _sixd_to_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64).reshape(6)
    r1, r2 = value[:3], value[3:]
    r1 = r1 / max(np.linalg.norm(r1), 1e-6)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / max(np.linalg.norm(r2), 1e-6)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=-1)


def _project_translation(
    centers_normalized: np.ndarray,
    log_depth: np.ndarray,
    intrinsic: object,
    image_shape: tuple[int, int],
    z_is_log: bool,
) -> np.ndarray:
    height, width = image_shape
    centers_px = np.asarray(centers_normalized, dtype=np.float64) * [width, height]
    centers_h = np.concatenate((centers_px, np.ones((len(centers_px), 1))), axis=1)
    z_values = np.asarray(log_depth, dtype=np.float64).reshape(-1, 1)
    depths = np.exp(z_values) if z_is_log else z_values
    return depths * (np.linalg.inv(intrinsic_matrix(intrinsic)) @ centers_h.T).T


def _draw_cuboid(
    canvas: np.ndarray,
    size: np.ndarray,
    transform: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    projected = project_corners(cuboid_corners(size, transform), intrinsic)
    points = np.rint(projected).astype(np.int32)
    for start, end in EDGES:
        cv2.line(
            canvas,
            tuple(points[start]),
            tuple(points[end]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _render_diagnostic_overlay(
    sample,
    post_prediction,
    matching,
    output_path: Path,
) -> None:
    image_path = Path(sample.metainfo["img_path"])
    canvas = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise FileNotFoundError(f"cannot decode RGB image: {image_path}")
    k = intrinsic_matrix(sample.metainfo["intrinsic"])
    gt = sample.gt_instances
    for size, transform in zip(_to_numpy(gt.sizes), _to_numpy(gt.T)):
        _draw_cuboid(canvas, size, transform, k, (0, 255, 0), 2)

    centers = _to_numpy(post_prediction.centers_2d)
    scores = _to_numpy(post_prediction.scores)
    for center, score in zip(centers, scores):
        fraction = float(np.clip((score - 0.4) / 0.15, 0.0, 1.0))
        color = (int(255 * (1 - fraction)), int(255 * fraction), 255)
        cv2.circle(canvas, tuple(np.rint(center).astype(int)), 2, color, -1)

    pred = matching["pred_instances"]
    assigned = _to_numpy(matching["assign_result"].gt_inds).astype(int)
    for query_index in np.flatnonzero(assigned > 0):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _sixd_to_matrix(_to_numpy(pred.rotations[query_index]))
        transform[:3, 3] = _to_numpy(pred.translations[query_index])
        try:
            _draw_cuboid(
                canvas,
                _to_numpy(pred.sizes[query_index]),
                transform,
                k,
                (0, 255, 255),
                1,
            )
        except (ValueError, FloatingPointError):
            pass
    cv2.putText(
        canvas,
        "GT=green matched=yellow queries=points",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write overlay: {output_path}")


def _aggregate_summaries(values: dict[str, list[float]]) -> dict:
    return {name: summarize_values(np.asarray(items)) for name, items in values.items()}


def main() -> None:
    args = parse_args()
    if args.max_images < 1 or args.overlay_images < 0:
        raise ValueError("--max-images must be positive and --overlay-images non-negative")
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("config and checkpoint must both exist")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    register_all_modules()
    register_mmengine_checkpoint_safe_globals()
    cfg = Config.fromfile(str(config_path))
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(output_dir / "runner")
    cfg.default_hooks.pop("checkpoint", None)
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(str(checkpoint_path))
    model = runner.model
    model.eval()
    head = model.bbox_head

    aggregate_costs: dict[str, list[float]] = defaultdict(list)
    aggregate_matched_costs: dict[str, list[float]] = defaultdict(list)
    aggregate_losses: dict[str, list[float]] = defaultdict(list)
    decode_errors: list[float] = []
    center_spreads: list[float] = []
    score_values: list[float] = []
    sample_reports = []
    image_count = 0

    with torch.no_grad():
        for data_batch in runner.val_dataloader:
            processed = model.data_preprocessor(data_batch, training=False)
            inputs = processed["inputs"]
            samples = processed["data_samples"]
            outputs = model._forward(inputs, samples)
            primary = outputs[:6]
            batch_metas = [sample.metainfo for sample in samples]
            batch_gt = [sample.gt_instances for sample in samples]
            losses = head.loss_by_feat_single(
                *(item[-1] for item in primary),
                None,
                None,
                None,
                batch_gt_instances=batch_gt,
                batch_img_metas=batch_metas,
                pose_supervision=True,
            )
            for name, value in zip(LOSS_NAMES, losses):
                numeric = float(value.detach().float().cpu().item())
                aggregate_losses[name].append(numeric)

            post_predictions = head.predict_by_feat(
                *primary,
                batch_img_metas=batch_metas,
                rescale=True,
            )
            for batch_index, (sample, post_prediction) in enumerate(
                zip(samples, post_predictions)
            ):
                if image_count >= args.max_images:
                    break
                cls_scores, bboxes, centers, z_values, rotations, sizes = (
                    item[-1, batch_index] for item in primary
                )
                matching = head.matching_diagnostics(
                    cls_scores,
                    bboxes,
                    centers,
                    z_values,
                    rotations,
                    sizes,
                    sample.gt_instances,
                    sample.metainfo,
                )
                gt = sample.gt_instances
                geometry = evaluate_geometry_contract(
                    intrinsic=sample.metainfo["intrinsic"],
                    translations=_to_numpy(gt.translations),
                    transforms=_to_numpy(gt.T),
                    centers_2d=_to_numpy(gt.centers_2d),
                    log_z=_to_numpy(gt.z),
                    sizes=_to_numpy(gt.sizes),
                    z_is_log=head.use_log_z,
                )

                expected_t = _project_translation(
                    _to_numpy(centers),
                    _to_numpy(z_values),
                    sample.metainfo["intrinsic"],
                    tuple(sample.metainfo["img_shape"]),
                    head.use_log_z,
                )
                matching_t = _to_numpy(matching["pred_instances"].translations)
                per_query_decode_error = np.linalg.norm(expected_t - matching_t, axis=1)
                decode_errors.extend(per_query_decode_error.tolist())

                assigned = _to_numpy(matching["assign_result"].gt_inds).astype(int)
                matched_queries = np.flatnonzero(assigned > 0)
                component_report = {}
                matched_abs_total = np.zeros(len(matched_queries), dtype=np.float64)
                matched_values_by_name = {}
                for name, matrix in matching["component_costs"].items():
                    values = _to_numpy(matrix)
                    aggregate_costs[name].extend(values.reshape(-1).tolist())
                    selected = np.array(
                        [values[q, assigned[q] - 1] for q in matched_queries],
                        dtype=np.float64,
                    )
                    aggregate_matched_costs[name].extend(selected.tolist())
                    matched_values_by_name[name] = selected
                    matched_abs_total += np.abs(selected)
                    component_report[name] = {
                        "matrix": summarize_values(values),
                        "matched": summarize_values(selected),
                    }
                contribution = {
                    name: float(
                        np.mean(
                            np.divide(
                                np.abs(values),
                                matched_abs_total,
                                out=np.zeros_like(values),
                                where=matched_abs_total > 0,
                            )
                        )
                    )
                    if len(values)
                    else 0.0
                    for name, values in matched_values_by_name.items()
                }

                post_scores = _to_numpy(post_prediction.scores).reshape(-1)
                post_centers = _to_numpy(post_prediction.centers_2d).reshape(-1, 2)
                score_values.extend(post_scores.tolist())
                center_spread = float(np.linalg.norm(post_centers.std(axis=0)))
                center_spreads.append(center_spread)
                overlay_path = None
                if image_count < args.overlay_images:
                    overlay_path = output_dir / "overlays" / (
                        f"{image_count:03d}_{Path(sample.metainfo['img_path']).stem}.png"
                    )
                    _render_diagnostic_overlay(
                        sample, post_prediction, matching, overlay_path
                    )

                sample_reports.append(
                    {
                        "record_index": image_count,
                        "img_path": str(Path(sample.metainfo["img_path"]).resolve()),
                        "overlay_path": (
                            str(overlay_path.resolve()) if overlay_path else None
                        ),
                        "intrinsic": intrinsic_matrix(
                            sample.metainfo["intrinsic"]
                        ).tolist(),
                        "geometry_contract": geometry,
                        "query_count": int(len(cls_scores)),
                        "gt_count": int(len(gt)),
                        "matched_query_indices": matched_queries.tolist(),
                        "matched_gt_indices": [
                            int(assigned[index] - 1) for index in matched_queries
                        ],
                        "matching_translation_decode_error_m": summarize_values(
                            per_query_decode_error
                        ),
                        "score": summarize_values(post_scores),
                        "center_spread_px": center_spread,
                        "component_costs": component_report,
                        "matched_cost_contribution_fraction": contribution,
                    }
                )
                image_count += 1
            if image_count >= args.max_images:
                break

    loss_summary = _aggregate_summaries(aggregate_losses)
    cost_summary = _aggregate_summaries(aggregate_costs)
    matched_cost_summary = _aggregate_summaries(aggregate_matched_costs)
    all_costs_finite = all(
        value["count"] == value["finite_count"] for value in cost_summary.values()
    )
    all_losses_finite = all(
        value["count"] == value["finite_count"] for value in loss_summary.values()
    )
    gt_contract_pass = all(
        sample["geometry_contract"]["pass"] for sample in sample_reports
    )
    decode_summary = summarize_values(decode_errors)
    decode_pass = bool(
        decode_summary["max"] is not None and decode_summary["max"] <= 1e-5
    )
    dominance = {}
    matched_names = list(aggregate_matched_costs)
    if matched_names:
        length = min(len(aggregate_matched_costs[name]) for name in matched_names)
        if length:
            arrays = {
                name: np.abs(np.asarray(aggregate_matched_costs[name][:length]))
                for name in matched_names
            }
            total = np.sum(list(arrays.values()), axis=0)
            dominance = {
                name: float(
                    np.mean(
                        np.divide(
                            values,
                            total,
                            out=np.zeros_like(values),
                            where=total > 0,
                        )
                    )
                )
                for name, values in arrays.items()
            }

    report = {
        "schema_version": 1,
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "camera_convention": (
            "T maps object-frame metres to camera-frame metres; z follows the "
            "configured log/raw encoding; "
            "centers_2d are pixels at the assigner boundary"
        ),
        "images_processed": image_count,
        "aggregate": {
            "geometry_contract_pass": gt_contract_pass,
            "matching_translation_decode_error_m": decode_summary,
            "component_costs": cost_summary,
            "matched_component_costs": matched_cost_summary,
            "matched_cost_contribution_fraction": dominance,
            "losses": loss_summary,
            "scores": summarize_values(score_values),
            "center_spread_px": summarize_values(center_spreads),
        },
        "gate": {
            "pass": bool(
                gt_contract_pass
                and all_costs_finite
                and all_losses_finite
                and decode_pass
            ),
            "gt_geometry": gt_contract_pass,
            "matching_translation_pixel_decode": decode_pass,
            "costs_finite": all_costs_finite,
            "losses_finite": all_losses_finite,
            "dominant_component_over_80_percent": any(
                fraction > 0.8 for fraction in dominance.values()
            ),
        },
        "samples": sample_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {report_path.resolve()}")
    print(json.dumps(report["gate"], sort_keys=True))
    if not report["gate"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
