#!/usr/bin/env python3
"""Measure query-level predicted OBB axes against matched custom-fruit GT."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.analysis_tools.compare_rgbd_rotation_predictions import (
    _label_path,
    pairwise_iou_xyxy,
)


SOURCE_SIZE = np.array([800.0, 600.0], dtype=np.float64)


def normalized_rboxes_to_compact_gaussians(
        rboxes: np.ndarray) -> np.ndarray:
    """Convert source 800x600 OBBs to normalized compact covariances."""
    rboxes = np.asarray(rboxes, dtype=np.float64)
    if rboxes.ndim != 2 or rboxes.shape[1] != 5:
        raise ValueError(f"rboxes must have shape (N, 5), got {rboxes.shape}")
    if (not np.isfinite(rboxes).all() or
            np.any(rboxes[:, 2:4] <= 0.0)):
        raise ValueError("rboxes must be finite with positive width/height")
    angles = rboxes[:, 4]
    cosine, sine = np.cos(angles), np.sin(angles)
    rotations = np.stack(
        (cosine, -sine, sine, cosine), axis=-1).reshape(-1, 2, 2)
    radii_squared = np.square(rboxes[:, 2:4] * 0.5)
    covariance = rotations @ np.apply_along_axis(
        np.diag, 1, radii_squared) @ np.swapaxes(rotations, -1, -2)
    normalization = np.diag(1.0 / SOURCE_SIZE)
    covariance = normalization @ covariance @ normalization
    return np.stack(
        (np.zeros(len(rboxes)), np.zeros(len(rboxes)),
         covariance[:, 0, 0], covariance[:, 0, 1],
         covariance[:, 1, 1]),
        axis=-1)


def _expand(compact: np.ndarray) -> np.ndarray:
    compact = np.asarray(compact, dtype=np.float64)
    if compact.ndim != 2 or compact.shape[1] != 5:
        raise ValueError(
            f"compact Gaussians must have shape (N, 5), got {compact.shape}")
    return np.stack(
        (compact[:, 2], compact[:, 3], compact[:, 3], compact[:, 4]),
        axis=-1).reshape(-1, 2, 2)


def _orientation_descriptor(compact: np.ndarray) -> np.ndarray:
    trace = compact[:, 2] + compact[:, 4]
    if not np.isfinite(compact).all() or np.any(trace <= 0.0):
        raise ValueError("compact covariance must be finite with positive trace")
    return np.stack(
        ((compact[:, 2] - compact[:, 4]) / trace,
         2.0 * compact[:, 3] / trace), axis=-1)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def _normalized_covariance_gwd(
        predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    first, second = _expand(predicted), _expand(target)
    trace_product = np.trace(first @ second, axis1=-2, axis2=-1)
    determinant_product = np.linalg.det(first) * np.linalg.det(second)
    covariance_distance = (
        np.trace(first, axis1=-2, axis2=-1)
        + np.trace(second, axis1=-2, axis2=-1)
        - 2.0 * np.sqrt(np.clip(
            trace_product + 2.0 * np.sqrt(
                np.clip(determinant_product, 0.0, None)),
            0.0, None)))
    distance = np.sqrt(np.clip(covariance_distance, 0.0, None))
    scale = 2.0 * np.clip(determinant_product, 1e-28, None) ** 0.125
    return distance / scale


def summarize_obb_alignment(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    bin_edges: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0),
) -> dict:
    """Summarize spin-2 alignment, axial error and anisotropy calibration."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 5:
        raise ValueError(
            f"predicted/target must share shape (N, 5), got "
            f"{predicted.shape}/{target.shape}")
    predicted_q = _orientation_descriptor(predicted)
    target_q = _orientation_descriptor(target)
    predicted_anisotropy = np.linalg.norm(predicted_q, axis=1)
    target_anisotropy = np.linalg.norm(target_q, axis=1)
    denominator = predicted_anisotropy * target_anisotropy
    observable = denominator > 1e-8
    alignment = np.full(len(target), np.nan, dtype=np.float64)
    alignment[observable] = np.sum(
        predicted_q[observable] * target_q[observable], axis=1
    ) / denominator[observable]
    alignment = np.clip(alignment, -1.0, 1.0)
    axis_error = np.degrees(np.arccos(alignment[observable])) * 0.5
    gwd = _normalized_covariance_gwd(predicted, target)
    if len(target) > 1 and np.std(predicted_anisotropy) > 0.0 and \
            np.std(target_anisotropy) > 0.0:
        correlation = float(np.corrcoef(
            predicted_anisotropy, target_anisotropy)[0, 1])
        if abs(abs(correlation) - 1.0) < 1e-12:
            correlation = float(np.sign(correlation))
    else:
        correlation = None
    weighted_alignment = float(np.nansum(
        alignment * target_anisotropy) /
        max(np.sum(target_anisotropy[observable]), 1e-12))

    edges = np.asarray(bin_edges, dtype=np.float64)
    if (len(edges) < 2 or not np.all(np.diff(edges) > 0.0) or
            (len(target) and (edges[0] > target_anisotropy.min() or
                              edges[-1] < target_anisotropy.max()))):
        raise ValueError("bin edges must increase and cover target anisotropy")
    bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        final_bin = index == len(edges) - 2
        selected = ((target_anisotropy >= lower) &
                    (target_anisotropy <= upper if final_bin
                     else target_anisotropy < upper))
        selected_observable = selected & observable
        item = {
            "lower_inclusive": float(lower),
            "upper_exclusive": None if final_bin else float(upper),
            "upper_inclusive": float(upper) if final_bin else None,
            "count": int(selected.sum()),
            "target_anisotropy": _summary(target_anisotropy[selected]),
            "predicted_anisotropy": _summary(
                predicted_anisotropy[selected]),
            "normalized_gwd": _summary(gwd[selected]),
            "double_angle_alignment": _summary(
                alignment[selected_observable]),
            "axis_error_degrees": _summary(
                np.degrees(np.arccos(
                    alignment[selected_observable])) * 0.5),
        }
        bins.append(item)
    return {
        "count": int(len(target)),
        "target_anisotropy": _summary(target_anisotropy),
        "predicted_anisotropy": _summary(predicted_anisotropy),
        "anisotropy_correlation": correlation,
        "weighted_double_angle_alignment": weighted_alignment,
        "double_angle_alignment": _summary(alignment[observable]),
        "axis_error_degrees": _summary(axis_error),
        "normalized_gwd": _summary(gwd),
        "target_anisotropy_bins": bins,
    }


def _matched_gaussians(
        prediction_dump: list[dict], iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    predicted_all, target_all = [], []
    for prediction in prediction_dump:
        with _label_path(prediction["img_path"]).open("rb") as handle:
            target = pickle.load(handle)
        instances = prediction["pred_instances"]
        if "obb_gaussians" not in instances:
            raise KeyError(
                "prediction dump lacks obb_gaussians; use a diagnostic config")
        predicted_boxes = np.asarray(instances["bboxes"], dtype=np.float64)
        target_boxes_yxyx = np.asarray(target["bboxes"], dtype=np.float64)
        target_boxes = target_boxes_yxyx[:, [1, 0, 3, 2]]
        ious = pairwise_iou_xyxy(predicted_boxes, target_boxes)
        pred_indices, gt_indices = linear_sum_assignment(-ious)
        accepted = ious[pred_indices, gt_indices] >= iou_threshold
        predicted_all.append(np.asarray(
            instances["obb_gaussians"], dtype=np.float64)[pred_indices[accepted]])
        target_compact = normalized_rboxes_to_compact_gaussians(
            target["obb_cxcywha_rad"])
        target_all.append(target_compact[gt_indices[accepted]])
    return np.concatenate(predicted_all), np.concatenate(target_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="named diagnostic dump in NAME=PATH form")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "contract": {
            "matching": "one-to-one Hungarian on 2D xyxy IoU",
            "iou_threshold": args.iou_threshold,
            "obb_coordinates": "covariance normalized from source 800x600",
            "axis_error": "0.5*acos(cos(2*delta_theta)), degrees",
        },
        "runs": {},
    }
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"prediction must be NAME=PATH, got {item!r}")
        with Path(path).open("rb") as handle:
            prediction_dump = pickle.load(handle)
        predicted, target = _matched_gaussians(
            prediction_dump, args.iou_threshold)
        report["runs"][name] = summarize_obb_alignment(predicted, target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
