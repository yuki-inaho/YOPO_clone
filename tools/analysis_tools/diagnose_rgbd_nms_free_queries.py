#!/usr/bin/env python3
"""Audit score quality and NMS dependence of RGB-D query prediction dumps.

The prediction dump remains the source of truth: NMS is evaluated only as a
counterfactual diagnostic and never changes the saved query-aligned 2D/3D data.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr


def _array(value: object) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def threshold_key(value: float) -> str:
    """Serialize a threshold without merging distinct sweep conditions."""
    return f"{float(value):.12g}"


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall, or zero at 0/0."""
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def _validate_thresholds(values: tuple[float, ...], name: str) -> None:
    if not values or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


def pairwise_iou_xyxy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two finite ``xyxy`` arrays."""
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=-1)
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0.0, None), axis=-1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0.0, None), axis=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return original query indices retained by greedy class-agnostic NMS."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        overlaps = pairwise_iou_xyxy(
            boxes[current:current + 1], boxes[remaining]).reshape(-1)
        order = remaining[overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary and aligned with scores")
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both positive and negative samples")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _match_count_and_ious(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if not len(pred_boxes) or not len(gt_boxes):
        return np.empty(0, dtype=np.float64)
    ious = pairwise_iou_xyxy(pred_boxes, gt_boxes)
    pred_indices, gt_indices = linear_sum_assignment(-ious)
    aligned = ious[pred_indices, gt_indices]
    return aligned[aligned >= threshold]


def summarize_selected_predictions(
    records: list[dict[str, np.ndarray]],
    *,
    score_threshold: float,
    nms_iou_threshold: float | None,
    match_iou_threshold: float,
) -> dict[str, float | int]:
    """Summarize selected queries under one threshold/NMS counterfactual."""
    kept_count = 0
    gt_count = 0
    matched_ious: list[np.ndarray] = []
    for record in records:
        boxes = record["boxes"]
        scores = record["scores"]
        selected = np.flatnonzero(scores >= score_threshold)
        if nms_iou_threshold is not None and selected.size:
            local_kept = nms_xyxy(
                boxes[selected], scores[selected], nms_iou_threshold)
            selected = selected[local_kept]
        kept_count += int(selected.size)
        gt_count += int(len(record["gt_boxes"]))
        matched_ious.append(_match_count_and_ious(
            boxes[selected], record["gt_boxes"], match_iou_threshold))
    accepted = (
        np.concatenate(matched_ious) if matched_ious
        else np.empty(0, dtype=np.float64)
    )
    true_positives = int(accepted.size)
    precision = true_positives / kept_count if kept_count else 0.0
    recall = true_positives / gt_count if gt_count else 0.0
    return {
        "kept": kept_count,
        "mean_kept_per_image": kept_count / len(records),
        "true_positives": true_positives,
        "ground_truths": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_matched_iou": float(accepted.mean()) if accepted.size else 0.0,
    }


def select_best_f1_selection(
    candidates: list[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    """Select one candidate using a complete, deterministic quality ordering.

    Higher F1, recall, and precision win, in that order. Remaining ties prefer
    a lower score threshold, no NMS, then a lower numeric NMS IoU threshold.
    """
    if not candidates:
        raise ValueError("at least one selection candidate is required")

    def selection_key(
        candidate: dict[str, float | int | None],
    ) -> tuple[float, float, float, float, bool, float]:
        nms_threshold = candidate["nms_iou_threshold"]
        return (
            float(candidate["f1"]),
            float(candidate["recall"]),
            float(candidate["precision"]),
            -float(candidate["score_threshold"]),
            nms_threshold is None,
            -float(nms_threshold) if nms_threshold is not None else 0.0,
        )

    return max(candidates, key=selection_key)


def _label_path(image_path: str) -> Path:
    suffix = "_color.png"
    if not image_path.endswith(suffix):
        raise ValueError(f"unexpected RGB path: {image_path}")
    return Path(image_path[:-len(suffix)] + "_label.pkl")


def load_records(path: Path) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Load a dump and its GT labels into compact NumPy records."""
    with path.open("rb") as stream:
        dump = pickle.load(stream)
    records = []
    image_paths = []
    for prediction in dump:
        image_path = prediction["img_path"]
        with _label_path(image_path).open("rb") as stream:
            target = pickle.load(stream)
        pred = prediction["pred_instances"]
        # NOCS label pickles store boxes as y1,x1,y2,x2.
        gt_yxyx = _array(target["bboxes"]).astype(np.float64)
        records.append({
            "boxes": _array(pred["bboxes"]).astype(np.float64),
            "scores": _array(pred["scores"]).astype(np.float64),
            "gt_boxes": gt_yxyx[:, [1, 0, 3, 2]],
        })
        image_paths.append(image_path)
    return records, image_paths


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }


def summarize_raw_scores(
    records: list[dict[str, np.ndarray]],
    match_iou_threshold: float,
) -> dict:
    """Measure whether raw query scores rank localized one-to-one positives."""
    labels_all = []
    scores_all = []
    best_ious_all = []
    for record in records:
        boxes = record["boxes"]
        gt_boxes = record["gt_boxes"]
        scores = record["scores"]
        ious = pairwise_iou_xyxy(boxes, gt_boxes)
        best_ious = ious.max(axis=1) if len(gt_boxes) else np.zeros(len(boxes))
        labels = np.zeros(len(boxes), dtype=np.int64)
        if len(boxes) and len(gt_boxes):
            pred_indices, gt_indices = linear_sum_assignment(-ious)
            accepted = ious[pred_indices, gt_indices] >= match_iou_threshold
            labels[pred_indices[accepted]] = 1
        labels_all.append(labels)
        scores_all.append(scores)
        best_ious_all.append(best_ious)
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    best_ious = np.concatenate(best_ious_all)
    correlation = spearmanr(scores, best_ious).statistic
    return {
        "queries": int(len(labels)),
        "positive_queries": int(labels.sum()),
        "negative_queries": int((labels == 0).sum()),
        "positive_score": _quantiles(scores[labels == 1]),
        "negative_score": _quantiles(scores[labels == 0]),
        "positive_vs_negative_roc_auc": binary_roc_auc(labels, scores),
        "score_best_iou_spearman": float(correlation),
    }


def summarize_dataset(records: list[dict[str, np.ndarray]]) -> dict:
    gt_counts = np.asarray([len(record["gt_boxes"]) for record in records])
    query_counts = np.asarray([len(record["boxes"]) for record in records])
    overlap_pairs = {"iou_gt_0.3": 0, "iou_gt_0.5": 0, "iou_gt_0.7": 0}
    for record in records:
        boxes = record["gt_boxes"]
        if len(boxes) < 2:
            continue
        ious = pairwise_iou_xyxy(boxes, boxes)
        upper = ious[np.triu_indices(len(boxes), k=1)]
        for threshold in (0.3, 0.5, 0.7):
            overlap_pairs[f"iou_gt_{threshold:.1f}"] += int((upper > threshold).sum())
    return {
        "images": int(len(records)),
        "ground_truths": int(gt_counts.sum()),
        "objects_per_image": {
            "median": float(np.median(gt_counts)),
            "p95": float(np.quantile(gt_counts, 0.95)),
            "max": int(gt_counts.max()),
        },
        "queries_per_image": {
            "min": int(query_counts.min()),
            "max": int(query_counts.max()),
        },
        "images_exceeding_query_capacity": int((gt_counts > query_counts).sum()),
        "gt_overlap_pair_counts": overlap_pairs,
    }


def diagnose_dump(
    records: list[dict[str, np.ndarray]],
    *,
    score_thresholds: tuple[float, ...],
    nms_iou_thresholds: tuple[float, ...],
    match_iou_threshold: float,
) -> dict:
    _validate_thresholds(score_thresholds, "score-thresholds")
    _validate_thresholds(nms_iou_thresholds, "nms-iou-thresholds")
    _validate_thresholds((match_iou_threshold,), "match-iou-threshold")
    selections = {}
    candidates = []
    for score_threshold in score_thresholds:
        per_nms = {
            "none": summarize_selected_predictions(
                records,
                score_threshold=score_threshold,
                nms_iou_threshold=None,
                match_iou_threshold=match_iou_threshold,
            )
        }
        candidates.append({
            "score_threshold": score_threshold,
            "nms_iou_threshold": None,
            **{
                key: per_nms["none"][key]
                for key in ("precision", "recall", "f1", "kept", "true_positives")
            },
        })
        for nms_threshold in nms_iou_thresholds:
            per_nms[threshold_key(nms_threshold)] = summarize_selected_predictions(
                records,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_threshold,
                match_iou_threshold=match_iou_threshold,
            )
            candidates.append({
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                **{
                    key: per_nms[threshold_key(nms_threshold)][key]
                    for key in (
                        "precision", "recall", "f1", "kept", "true_positives"
                    )
                },
            })
        selections[threshold_key(score_threshold)] = per_nms
    return {
        "raw_score_alignment": summarize_raw_scores(
            records, match_iou_threshold),
        "selection_counterfactuals": selections,
        "best_f1_selection": select_best_f1_selection(candidates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="named prediction dump in NAME=PATH form")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-thresholds", type=float, nargs="+", default=(0.2, 0.3, 0.4, 0.5))
    parser.add_argument(
        "--nms-iou-thresholds", type=float, nargs="+", default=(0.3, 0.5, 0.7))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_thresholds(tuple(args.score_thresholds), "score-thresholds")
    _validate_thresholds(tuple(args.nms_iou_thresholds), "nms-iou-thresholds")
    _validate_thresholds((args.match_iou_threshold,), "match-iou-threshold")

    named_paths: dict[str, Path] = {}
    for item in args.prediction:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in named_paths:
            raise ValueError(f"prediction must be unique NAME=PATH, got {item!r}")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        named_paths[name] = path

    loaded = {name: load_records(path) for name, path in named_paths.items()}
    reference_paths = next(iter(loaded.values()))[1]
    for name, (_, image_paths) in loaded.items():
        if image_paths != reference_paths:
            raise ValueError(f"image ordering differs for prediction {name!r}")
    reference_records = next(iter(loaded.values()))[0]
    report = {
        "schema_version": 1,
        "contract": {
            "matching": "one-to-one Hungarian on 2D xyxy IoU",
            "match_iou_threshold": args.match_iou_threshold,
            "nms": "counterfactual diagnostic only; raw dump and inference graph unchanged",
            "query_identity": "retained indices refer to the same 2D/OBB/3D query",
            "best_f1_tie_break": (
                "higher F1, then recall, then precision; lower score threshold; "
                "no NMS; then lower numeric NMS IoU threshold"
            ),
        },
        "dataset": summarize_dataset(reference_records),
        "runs": {
            name: {
                "prediction": str(named_paths[name].resolve()),
                **diagnose_dump(
                    records,
                    score_thresholds=tuple(args.score_thresholds),
                    nms_iou_thresholds=tuple(args.nms_iou_thresholds),
                    match_iou_threshold=args.match_iou_threshold,
                ),
            }
            for name, (records, _) in loaded.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
