#!/usr/bin/env python3
"""Evaluate matched RGB-D rotations under explicit local-axis symmetries."""

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
    rotation_error_degrees,
    summarize,
)


AXIS_NAMES = ("x", "y", "z")


def axial_quotient_error_degrees(
    predicted: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    unoriented: bool = False,
) -> np.ndarray:
    """SO(3)/SO(2) error from the angle between corresponding local axes."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    predicted_axis = np.asarray(predicted, dtype=np.float64)[..., :, axis]
    target_axis = np.asarray(target, dtype=np.float64)[..., :, axis]
    cosine = np.sum(predicted_axis * target_axis, axis=-1)
    if unoriented:
        cosine = np.abs(cosine)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    first, second = [index for index in range(3) if index != axis]
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[first, first] = matrix[second, second] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    return matrix


def half_turn_quotient_error_degrees(
        predicted: np.ndarray, target: np.ndarray, axis: int) -> np.ndarray:
    """Minimum geodesic error with/without a local-axis 180-degree turn."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    half_turn_target = target @ _axis_rotation(axis, np.pi)
    return np.minimum(
        rotation_error_degrees(predicted, target),
        rotation_error_degrees(predicted, half_turn_target))


def _matched_pose(
    prediction_dump: list[dict], iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_all, target_all, sizes_all = [], [], []
    for prediction in prediction_dump:
        with _label_path(prediction["img_path"]).open("rb") as handle:
            target = pickle.load(handle)
        instances = prediction["pred_instances"]
        predicted_boxes = np.asarray(instances["bboxes"], dtype=np.float64)
        target_boxes_yxyx = np.asarray(target["bboxes"], dtype=np.float64)
        target_boxes = target_boxes_yxyx[:, [1, 0, 3, 2]]
        ious = pairwise_iou_xyxy(predicted_boxes, target_boxes)
        pred_indices, gt_indices = linear_sum_assignment(-ious)
        accepted = ious[pred_indices, gt_indices] >= iou_threshold
        predicted_all.append(np.asarray(
            instances["T"], dtype=np.float64)[pred_indices[accepted], :3, :3])
        target_all.append(np.asarray(
            target["rotations"], dtype=np.float64)[gt_indices[accepted]])
        sizes = (np.asarray(target["sizes"], dtype=np.float64) *
                 np.asarray(target["scales"], dtype=np.float64)[:, None])
        sizes_all.append(sizes[gt_indices[accepted]])
    return (
        np.concatenate(predicted_all),
        np.concatenate(target_all),
        np.concatenate(sizes_all),
    )


def symmetry_report(
        predicted: np.ndarray, target: np.ndarray, sizes: np.ndarray) -> dict:
    """Summarize full SO(3), C2 and continuous axial quotients."""
    report = {"full_so3": summarize(
        rotation_error_degrees(predicted, target)), "axes": {}}
    for axis, name in enumerate(AXIS_NAMES):
        other = [index for index in range(3) if index != axis]
        relative_size_difference = np.abs(
            sizes[:, other[0]] - sizes[:, other[1]]) / np.clip(
                0.5 * (sizes[:, other[0]] + sizes[:, other[1]]), 1e-12, None)
        report["axes"][name] = {
            "orthogonal_size_relative_difference": summarize(
                relative_size_difference),
            "half_turn_c2": summarize(
                half_turn_quotient_error_degrees(predicted, target, axis)),
            "continuous_oriented_so2": summarize(
                axial_quotient_error_degrees(predicted, target, axis)),
            "continuous_unoriented_o2": summarize(
                axial_quotient_error_degrees(
                    predicted, target, axis, unoriented=True)),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="named prediction in NAME=PATH form")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "contract": {
            "matching": "one-to-one Hungarian on 2D xyxy IoU",
            "iou_threshold": args.iou_threshold,
            "half_turn": "min over identity and local-axis pi rotation",
            "continuous_oriented": "angle between oriented local axes",
            "continuous_unoriented": "angle between unoriented local axes",
        },
        "runs": {},
    }
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"prediction must be NAME=PATH, got {item!r}")
        with Path(path).open("rb") as handle:
            prediction_dump = pickle.load(handle)
        predicted, target, sizes = _matched_pose(
            prediction_dump, args.iou_threshold)
        report["runs"][name] = symmetry_report(predicted, target, sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
