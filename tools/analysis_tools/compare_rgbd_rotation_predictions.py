#!/usr/bin/env python3
"""Compare matched RGB-D rotation errors across prediction dumps."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_iou_xyxy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two ``xyxy`` arrays."""
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=-1)
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0.0, None), axis=-1)
    second_area = np.prod(
        np.clip(second[:, 2:] - second[:, :2], 0.0, None), axis=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def rotation_error_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) error in degrees for aligned matrix batches."""
    relative = np.swapaxes(predicted, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
                     -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def obb_anisotropy(rboxes: np.ndarray) -> np.ndarray:
    """Return scale/angle-invariant ellipse anisotropy from OBB width/height."""
    rboxes = np.asarray(rboxes, dtype=np.float64)
    if rboxes.ndim != 2 or rboxes.shape[1] != 5:
        raise ValueError(f"OBBs must have shape (N, 5), got {rboxes.shape}")
    squared_sizes = np.square(rboxes[:, 2:4])
    denominator = squared_sizes.sum(axis=1)
    if not np.isfinite(rboxes).all() or np.any(denominator <= 0.0):
        raise ValueError("OBBs must be finite with positive width and height")
    return np.abs(squared_sizes[:, 0] - squared_sizes[:, 1]) / denominator


def summarize_paired_by_anisotropy(
    first: np.ndarray,
    second: np.ndarray,
    anisotropy: np.ndarray,
    *,
    bin_edges: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0),
) -> list[dict[str, float | int | None]]:
    """Summarize paired rotation deltas in target-OBB observability bins."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    anisotropy = np.asarray(anisotropy, dtype=np.float64).reshape(-1)
    if len({len(first), len(second), len(anisotropy)}) != 1:
        raise ValueError("paired errors and anisotropy must have equal lengths")
    if not all(np.isfinite(value).all()
               for value in (first, second, anisotropy)):
        raise ValueError("paired errors and anisotropy must be finite")
    edges = np.asarray(bin_edges, dtype=np.float64)
    if (len(edges) < 2 or not np.all(np.diff(edges) > 0.0) or
            (len(anisotropy) and
             (edges[0] > anisotropy.min() or edges[-1] < anisotropy.max()))):
        raise ValueError("bin edges must be increasing and cover anisotropy")

    delta = second - first
    result = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        final_bin = index == len(edges) - 2
        selected = ((anisotropy >= lower) &
                    (anisotropy <= upper if final_bin else anisotropy < upper))
        item = {
            "lower_inclusive": float(lower),
            "upper_exclusive": None if final_bin else float(upper),
            "upper_inclusive": float(upper) if final_bin else None,
            "count": int(selected.sum()),
        }
        if np.any(selected):
            item.update({
                "mean_rotation_delta_degrees": float(delta[selected].mean()),
                "median_rotation_delta_degrees": float(
                    np.median(delta[selected])),
                "second_better_fraction": float(
                    np.mean(delta[selected] < 0.0)),
            })
        result.append(item)
    return result


def match_image(
    prediction: dict,
    target: dict,
    iou_threshold: float,
) -> dict[int, dict[str, float]]:
    """Match predictions to GT by 2D IoU and retain accepted GT identities."""
    predicted = prediction["pred_instances"]
    predicted_boxes = np.asarray(predicted["bboxes"], dtype=np.float64)
    # NOCS label pickles store boxes as y1,x1,y2,x2.
    target_boxes_yxyx = np.asarray(target["bboxes"], dtype=np.float64)
    target_boxes = target_boxes_yxyx[:, [1, 0, 3, 2]]
    target_anisotropy = obb_anisotropy(target["obb_cxcywha_rad"])
    ious = pairwise_iou_xyxy(predicted_boxes, target_boxes)
    pred_indices, gt_indices = linear_sum_assignment(-ious)

    predicted_rotations = np.asarray(predicted["T"], dtype=np.float64)[:, :3, :3]
    target_rotations = np.asarray(target["rotations"], dtype=np.float64)
    errors = rotation_error_degrees(
        predicted_rotations[pred_indices], target_rotations[gt_indices])
    scores = np.asarray(predicted["scores"], dtype=np.float64)[pred_indices]
    translations = np.asarray(predicted["translations"], dtype=np.float64)
    target_translations = np.asarray(target["translations"], dtype=np.float64)
    translation_errors = np.linalg.norm(
        translations[pred_indices] - target_translations[gt_indices], axis=-1)

    result = {}
    for pred_index, gt_index, error, score, translation_error in zip(
            pred_indices, gt_indices, errors, scores, translation_errors):
        iou = float(ious[pred_index, gt_index])
        if iou >= iou_threshold:
            result[int(gt_index)] = {
                "iou": iou,
                "score": float(score),
                "rotation_degrees": float(error),
                "translation_metres": float(translation_error),
                "target_obb_anisotropy": float(target_anisotropy[gt_index]),
            }
    return result


def summarize(values: np.ndarray) -> dict[str, float | int]:
    """Stable scalar summary for one error vector."""
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _label_path(image_path: str) -> Path:
    suffix = "_color.png"
    if not image_path.endswith(suffix):
        raise ValueError(f"unexpected RGB path: {image_path}")
    return Path(image_path[:-len(suffix)] + "_label.pkl")


def compare_prediction_dumps(
    named_paths: dict[str, Path],
    iou_threshold: float,
) -> dict:
    """Build per-run and paired matched-GT rotation summaries."""
    dumps = {}
    for name, path in named_paths.items():
        with path.open("rb") as handle:
            dumps[name] = pickle.load(handle)
    lengths = {name: len(value) for name, value in dumps.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction dump lengths differ: {lengths}")

    per_run: dict[str, dict[tuple[int, int], dict[str, float]]] = {
        name: {} for name in named_paths}
    for image_index in range(next(iter(lengths.values()))):
        image_paths = {
            name: dump[image_index]["img_path"] for name, dump in dumps.items()}
        if len(set(image_paths.values())) != 1:
            raise ValueError(
                f"image ordering differs at {image_index}: {image_paths}")
        with _label_path(next(iter(image_paths.values()))).open("rb") as handle:
            target = pickle.load(handle)
        for name, dump in dumps.items():
            matched = match_image(dump[image_index], target, iou_threshold)
            per_run[name].update({
                (image_index, gt_index): metrics
                for gt_index, metrics in matched.items()
            })

    report = {
        "contract": {
            "matching": "one-to-one Hungarian on 2D xyxy IoU",
            "iou_threshold": iou_threshold,
            "rotation_error": "SO(3) geodesic degrees, no symmetry quotient",
        },
        "runs": {},
        "paired": {},
    }
    for name, matched in per_run.items():
        rotation = np.asarray(
            [value["rotation_degrees"] for value in matched.values()])
        translation = np.asarray(
            [value["translation_metres"] for value in matched.values()])
        rotation_summary = summarize(rotation)
        rotation_summary.update({
            "within_5_degrees_fraction": float(np.mean(rotation <= 5.0)),
            "within_10_degrees_fraction": float(np.mean(rotation <= 10.0)),
            "within_30_degrees_fraction": float(np.mean(rotation <= 30.0)),
        })
        translation_summary = summarize(translation)
        translation_summary.update({
            "within_2cm_fraction": float(np.mean(translation <= 0.02)),
            "within_5cm_fraction": float(np.mean(translation <= 0.05)),
            "within_10cm_fraction": float(np.mean(translation <= 0.10)),
        })
        report["runs"][name] = {
            "rotation_degrees": rotation_summary,
            "translation_metres": translation_summary,
        }

    names = list(named_paths)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            common = sorted(set(per_run[first_name]) & set(per_run[second_name]))
            first = np.asarray([
                per_run[first_name][key]["rotation_degrees"] for key in common])
            second = np.asarray([
                per_run[second_name][key]["rotation_degrees"] for key in common])
            delta = second - first
            anisotropy = np.asarray([
                per_run[first_name][key]["target_obb_anisotropy"]
                for key in common])
            report["paired"][f"{second_name}_minus_{first_name}"] = {
                "count": int(delta.size),
                "mean_rotation_degrees": float(np.mean(delta)),
                "median_rotation_degrees": float(np.median(delta)),
                "second_better_fraction": float(np.mean(delta < 0.0)),
                "second_equal_fraction": float(np.mean(delta == 0.0)),
                "target_obb_anisotropy_bins":
                    summarize_paired_by_anisotropy(
                        first, second, anisotropy),
            }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="named dump in NAME=PATH form; provide at least two")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = {}
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"prediction must be NAME=PATH, got {item!r}")
        named_paths[name] = Path(path)
    if len(named_paths) < 2:
        raise ValueError("at least two uniquely named predictions are required")
    report = compare_prediction_dumps(named_paths, args.iou_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
