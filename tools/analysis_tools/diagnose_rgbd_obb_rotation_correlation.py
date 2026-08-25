#!/usr/bin/env python3
"""Audit whether observed 2D OBBs constrain custom-fruit 3D rotation.

The custom pkl stores OBBs in the source 800x600 coordinate system while the
converted RGB, axis-aligned boxes and camera intrinsic use 640x480 pixels.
This tool first estimates that coordinate conversion, then projects each 3D
ellipsoid exactly through its dual quadric and compares the resulting 2D
Gaussian with the observed OBB Gaussian using normalized GWD.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_INTRINSIC = np.array(
    [
        [443.9066, 0.0, 321.3503],
        [0.0, 449.1953, 230.8687],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
IDENTITY_PERMUTATION = (0, 1, 2)
SIZE_PERMUTATIONS = tuple(permutations(range(3)))


def canonical_camera_rotation() -> np.ndarray:
    """Return the generator's camera-anchored local frame.

    Local X points into camera depth, local Z points to the sideways rig's
    world-up direction (camera +X), and local Y completes a right-handed
    frame.
    """
    return np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )


def infer_obb_coordinate_scale(
    obb_centers: np.ndarray,
    bbox_centers: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit the shared least-squares scale and return per-instance residuals."""
    obb_centers = np.asarray(obb_centers, dtype=np.float64)
    bbox_centers = np.asarray(bbox_centers, dtype=np.float64)
    if obb_centers.shape != bbox_centers.shape or obb_centers.ndim != 2 or \
            obb_centers.shape[1] != 2:
        raise ValueError(
            "obb_centers and bbox_centers must both have shape (N, 2), got "
            f"{obb_centers.shape} and {bbox_centers.shape}")
    denominator = float(np.square(obb_centers).sum())
    if denominator <= 0.0:
        raise ValueError("cannot infer OBB scale from zero-valued centers")
    scale = float((obb_centers * bbox_centers).sum() / denominator)
    residuals = np.linalg.norm(obb_centers * scale - bbox_centers, axis=1)
    return scale, residuals


def infer_axis_scales(
    obb_centers: np.ndarray,
    bbox_centers: np.ndarray,
) -> np.ndarray:
    """Fit independent x/y scales for diagnostic reporting."""
    numerator = (obb_centers * bbox_centers).sum(axis=0)
    denominator = np.square(obb_centers).sum(axis=0)
    if np.any(denominator <= 0.0):
        raise ValueError("cannot infer per-axis OBB scale from zero centers")
    return numerator / denominator


def rbox_to_gaussian(rbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(cx, cy, full-width, full-height, radians)`` to Gaussian."""
    rbox = np.asarray(rbox, dtype=np.float64)
    if rbox.shape != (5,):
        raise ValueError(f"rbox must have shape (5,), got {rbox.shape}")
    if not np.isfinite(rbox).all() or np.any(rbox[2:4] <= 0.0):
        raise ValueError(f"rbox must be finite with positive sizes, got {rbox}")
    cosine = np.cos(rbox[4])
    sine = np.sin(rbox[4])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    sigma = rotation @ np.diag(np.square(rbox[2:4] * 0.5)) @ rotation.T
    return rbox[:2].copy(), sigma


def project_ellipsoid_gaussian(
    translation: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
    intrinsic: np.ndarray = DEFAULT_INTRINSIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly project a camera-frame ellipsoid through its dual quadric."""
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if not all(np.isfinite(value).all() for value in
               (translation, rotation, size, intrinsic)):
        raise ValueError("ellipsoid projection inputs must be finite")
    if np.any(size <= 0.0):
        raise ValueError(f"ellipsoid size must be positive, got {size}")
    if translation[2] <= np.max(size) * 0.5:
        raise ValueError(
            "camera must be outside and in front of the ellipsoid, got "
            f"z={translation[2]} and size={size}")

    radii_squared = np.square(size * 0.5)
    shape = rotation @ np.diag(radii_squared) @ rotation.T
    dual_conic = intrinsic @ (
        shape - np.outer(translation, translation)) @ intrinsic.T
    conic = np.linalg.inv(dual_conic)
    matrix = conic[:2, :2]
    vector = conic[:2, 2]
    matrix_inv = np.linalg.inv(matrix)
    center = -(matrix_inv @ vector)
    centered_constant = conic[2, 2] - vector @ matrix_inv @ vector
    sigma = -centered_constant * matrix_inv
    sigma = (sigma + sigma.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(sigma)
    if not np.isfinite(center).all() or not np.isfinite(sigma).all() or \
            np.any(eigenvalues <= 0.0):
        raise ValueError(
            "projected conic is not a finite ellipse: "
            f"center={center}, eigenvalues={eigenvalues}")
    return center, sigma


def gaussian_gwd_distance(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    normalize: bool = True,
) -> float:
    """Return the repo's raw 2x2 GWD distance before nonlinear postprocess."""
    xy_first, sigma_first = first
    xy_second, sigma_second = second
    xy_distance = float(np.square(xy_first - xy_second).sum())
    trace_product = float(np.trace(sigma_first @ sigma_second))
    determinant_product = float(
        np.linalg.det(sigma_first) * np.linalg.det(sigma_second))
    determinant_sqrt = np.sqrt(max(determinant_product, 0.0))
    covariance_distance = float(
        np.trace(sigma_first) + np.trace(sigma_second)
        - 2.0 * np.sqrt(max(trace_product + 2.0 * determinant_sqrt, 0.0)))
    distance = np.sqrt(max(xy_distance + covariance_distance, 0.0))
    if normalize:
        scale = 2.0 * max(determinant_product, 1e-28) ** 0.125
        distance /= scale
    return float(distance)


def gwd_distance(
    first_rbox: np.ndarray,
    second_rbox: np.ndarray,
    normalize: bool = True,
) -> float:
    return gaussian_gwd_distance(
        rbox_to_gaussian(first_rbox),
        rbox_to_gaussian(second_rbox),
        normalize=normalize,
    )


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def summarize_observability(
    paired: np.ndarray,
    canonical: np.ndarray,
    shuffled: np.ndarray,
    anisotropy: np.ndarray,
    *,
    bin_edges: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0),
) -> dict:
    """Summarize how target ellipse anisotropy exposes rotation signal."""
    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (paired, canonical, shuffled, anisotropy)
    )
    paired, canonical, shuffled, anisotropy = arrays
    if len({len(value) for value in arrays}) != 1 or len(anisotropy) == 0:
        raise ValueError(
            "observability arrays must be non-empty with equal lengths")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("observability arrays must be finite")
    edges = np.asarray(bin_edges, dtype=np.float64)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0.0) or \
            edges[0] > float(anisotropy.min()) or \
            edges[-1] < float(anisotropy.max()):
        raise ValueError(
            "bin_edges must be strictly increasing and cover anisotropy")

    gain = shuffled - paired
    correlation = float(np.corrcoef(anisotropy, gain)[0, 1])
    bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            selected = (anisotropy >= lower) & (anisotropy <= upper)
        else:
            selected = (anisotropy >= lower) & (anisotropy < upper)
        count = int(selected.sum())
        item = {
            "lower_inclusive": float(lower),
            "upper_exclusive": None if index == len(edges) - 2
            else float(upper),
            "upper_inclusive": float(upper) if index == len(edges) - 2
            else None,
            "count": count,
            "fraction": float(count / len(anisotropy)),
        }
        if count:
            paired_median = float(np.median(paired[selected]))
            canonical_median = float(np.median(canonical[selected]))
            shuffled_median = float(np.median(shuffled[selected]))
            item.update({
                "anisotropy": _summary(anisotropy[selected]),
                "paired_gwd": _summary(paired[selected]),
                "canonical_gwd": _summary(canonical[selected]),
                "shuffled_gwd": _summary(shuffled[selected]),
                "paired_to_shuffled_median_ratio": paired_median / max(
                    shuffled_median, 1e-12),
                "paired_to_canonical_median_ratio": paired_median / max(
                    canonical_median, 1e-12),
                "paired_better_than_shuffled_fraction": float(
                    np.mean(paired[selected] < shuffled[selected])),
            })
        bins.append(item)
    return {
        "definition": "(lambda_max-lambda_min)/(lambda_max+lambda_min)",
        "anisotropy": _summary(anisotropy),
        "correlation_with_shuffled_minus_paired": correlation,
        "bins": bins,
    }


def _gaussian_axis_alignment(
    predicted_sigma: np.ndarray,
    target_sigma: np.ndarray,
) -> tuple[float, float]:
    target_values, target_vectors = np.linalg.eigh(target_sigma)
    predicted_values, predicted_vectors = np.linalg.eigh(predicted_sigma)
    target_axis = target_vectors[:, int(np.argmax(target_values))]
    predicted_axis = predicted_vectors[:, int(np.argmax(predicted_values))]
    target_angle = np.arctan2(target_axis[1], target_axis[0])
    predicted_angle = np.arctan2(predicted_axis[1], predicted_axis[0])
    alignment = float(np.cos(2.0 * (predicted_angle - target_angle)))
    anisotropy = float(
        (target_values[-1] - target_values[0]) /
        max(target_values[-1] + target_values[0], 1e-12))
    return alignment, anisotropy


def _load_split(data_root: Path, list_name: str) -> dict[str, np.ndarray]:
    real_root = data_root / "real"
    translations = []
    rotations = []
    sizes = []
    rboxes = []
    bbox_centers = []
    for relative in (real_root / list_name).read_text().splitlines():
        if not relative:
            continue
        with (real_root / f"{relative}_label.pkl").open("rb") as stream:
            label = pickle.load(stream)
        boxes = np.asarray(label["bboxes"], dtype=np.float64)
        obbs = np.asarray(label["obb_cxcywha_rad"], dtype=np.float64)
        if len(boxes) != len(obbs):
            raise ValueError(
                f"bbox/OBB count mismatch in {relative}: {len(boxes)} vs {len(obbs)}")
        translations.extend(np.asarray(label["translations"], dtype=np.float64))
        rotations.extend(np.asarray(label["rotations"], dtype=np.float64))
        sizes.extend(
            np.asarray(label["sizes"], dtype=np.float64)
            * np.asarray(label["scales"], dtype=np.float64)[:, None])
        rboxes.extend(obbs)
        bbox_centers.extend(
            np.stack(
                ((boxes[:, 1] + boxes[:, 3]) * 0.5,
                 (boxes[:, 0] + boxes[:, 2]) * 0.5),
                axis=1,
            ))
    return {
        "translations": np.asarray(translations),
        "rotations": np.asarray(rotations),
        "sizes": np.asarray(sizes),
        "rboxes": np.asarray(rboxes),
        "bbox_centers": np.asarray(bbox_centers),
    }


def _projection_distances(
    split: dict[str, np.ndarray],
    obb_scale: float,
    size_permutation: tuple[int, int, int],
    rotation_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = split["rotations"]
    if rotation_mode == "canonical":
        rotations = np.broadcast_to(
            canonical_camera_rotation(), rotations.shape)
    elif rotation_mode == "shuffled":
        rotations = np.roll(rotations, 1, axis=0)
    elif rotation_mode != "paired":
        raise ValueError(f"unsupported rotation mode: {rotation_mode}")

    distances = []
    alignments = []
    anisotropies = []
    for translation, rotation, size, raw_rbox in zip(
        split["translations"], rotations, split["sizes"], split["rboxes"]
    ):
        target_rbox = raw_rbox.copy()
        target_rbox[:4] *= obb_scale
        target = rbox_to_gaussian(target_rbox)
        predicted = project_ellipsoid_gaussian(
            translation,
            rotation,
            size[list(size_permutation)],
        )
        distances.append(gaussian_gwd_distance(predicted, target))
        alignment, anisotropy = _gaussian_axis_alignment(
            predicted[1], target[1])
        alignments.append(alignment)
        anisotropies.append(anisotropy)
    return (
        np.asarray(distances),
        np.asarray(alignments),
        np.asarray(anisotropies),
    )


def _analyze_split_scale(split: dict[str, np.ndarray]) -> dict:
    obb_centers = split["rboxes"][:, :2]
    scale, residuals = infer_obb_coordinate_scale(
        obb_centers, split["bbox_centers"])
    axis_scales = infer_axis_scales(obb_centers, split["bbox_centers"])
    return {
        "shared_scale": scale,
        "axis_scales": axis_scales.tolist(),
        "center_residual_px": _summary(residuals),
    }


def _permutation_report(
    split: dict[str, np.ndarray],
    obb_scale: float,
) -> dict[str, dict]:
    report = {}
    for permutation in SIZE_PERMUTATIONS:
        distances, _, _ = _projection_distances(
            split, obb_scale, permutation, "paired")
        report["".join(map(str, permutation))] = _summary(distances)
    return report


def _mode_report(
    split: dict[str, np.ndarray],
    obb_scale: float,
    size_permutation: tuple[int, int, int],
) -> dict:
    raw = {}
    distances_by_mode = {}
    target_anisotropies = None
    for mode in ("paired", "canonical", "shuffled"):
        distances, alignments, anisotropies = _projection_distances(
            split, obb_scale, size_permutation, mode)
        weighted_alignment = float(
            np.sum(alignments * anisotropies) /
            max(np.sum(anisotropies), 1e-12))
        raw[mode] = {
            "gwd": _summary(distances),
            "weighted_double_angle_alignment": weighted_alignment,
        }
        distances_by_mode[mode] = distances
        if target_anisotropies is None:
            target_anisotropies = anisotropies
        elif not np.allclose(target_anisotropies, anisotropies):
            raise RuntimeError("target anisotropy changed across rotation modes")
    paired = distances_by_mode["paired"]
    canonical = distances_by_mode["canonical"]
    shuffled = distances_by_mode["shuffled"]
    raw["comparisons"] = {
        "paired_better_than_shuffled_fraction": float(
            np.mean(paired < shuffled)),
        "paired_better_than_canonical_fraction": float(
            np.mean(paired < canonical)),
        "paired_to_shuffled_median_ratio": float(
            np.median(paired) / max(np.median(shuffled), 1e-12)),
        "paired_to_canonical_median_ratio": float(
            np.median(paired) / max(np.median(canonical), 1e-12)),
    }
    raw["observability"] = summarize_observability(
        paired, canonical, shuffled, target_anisotropies)
    return raw


def build_report(data_root: Path) -> dict:
    train = _load_split(data_root, "train_list.txt")
    val = _load_split(data_root, "test_list.txt")
    train_scale = _analyze_split_scale(train)
    val_scale = _analyze_split_scale(val)
    obb_scale = float(train_scale["shared_scale"])

    train_permutations = _permutation_report(train, obb_scale)
    val_permutations = _permutation_report(val, obb_scale)
    train_best_key = min(
        train_permutations,
        key=lambda key: train_permutations[key]["median"],
    )
    val_best_key = min(
        val_permutations,
        key=lambda key: val_permutations[key]["median"],
    )
    identity_key = "012"
    improvement = 1.0 - (
        train_permutations[train_best_key]["median"] /
        train_permutations[identity_key]["median"])
    stable_alternative = (
        train_best_key == val_best_key
        and train_best_key != identity_key
        and improvement >= 0.10
    )
    selected_key = train_best_key if stable_alternative else identity_key
    selected_permutation = tuple(int(value) for value in selected_key)

    train_modes = _mode_report(
        train, obb_scale, selected_permutation)
    val_modes = _mode_report(val, obb_scale, selected_permutation)
    val_comparison = val_modes["comparisons"]
    scale_gate = (
        abs(train_scale["axis_scales"][0] - 0.8) <= 1e-4
        and abs(train_scale["axis_scales"][1] - 0.8) <= 1e-4
        and abs(val_scale["axis_scales"][0] - 0.8) <= 1e-4
        and abs(val_scale["axis_scales"][1] - 0.8) <= 1e-4
        and train_scale["center_residual_px"]["p99"] <= 1e-3
        and val_scale["center_residual_px"]["p99"] <= 1e-3
    )
    rotation_gate = (
        val_comparison["paired_to_shuffled_median_ratio"] <= 0.90
        and val_comparison["paired_better_than_shuffled_fraction"] >= 0.60
        and val_comparison["paired_to_canonical_median_ratio"] <= 0.95
    )
    return {
        "schema_version": 2,
        "data_root": str(data_root.resolve()),
        "intrinsic": DEFAULT_INTRINSIC.tolist(),
        "counts": {"train": len(train["sizes"]), "val": len(val["sizes"])},
        "obb_coordinate_contract": {
            "train": train_scale,
            "val": val_scale,
            "training_scale_used": obb_scale,
            "gate_passed": bool(scale_gate),
        },
        "size_axis_permutation": {
            "train_candidates": train_permutations,
            "val_candidates": val_permutations,
            "train_best": train_best_key,
            "val_best": val_best_key,
            "train_best_improvement_over_stored": improvement,
            "stable_alternative": stable_alternative,
            "selected": selected_key,
        },
        "projection_consistency": {
            "train": train_modes,
            "val": val_modes,
            "rotation_information_gate_passed": bool(rotation_gate),
        },
        "all_pretraining_gates_passed": bool(scale_gate and rotation_gate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/nocs_custom"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": report["counts"],
        "obb_scale": report["obb_coordinate_contract"]["training_scale_used"],
        "selected_size_permutation": report["size_axis_permutation"]["selected"],
        "val_comparisons": report["projection_consistency"]["val"]["comparisons"],
        "all_pretraining_gates_passed": report["all_pretraining_gates_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
