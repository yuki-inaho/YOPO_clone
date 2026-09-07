"""Pure contracts for transferring pseudo supervision to detector predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .online_tracker import partial_linear_assignment

PREDICTION_FEATURE_CACHE_SCHEMA = "yopo_g10_prediction_sequence_features_v1"


@dataclass(frozen=True)
class TeacherAssignment:
    """One-to-one teacher metadata aligned to retained prediction rows."""

    teacher_indices: Tensor
    ious: Tensor
    center_distances_px: Tensor
    matched_prediction_indices: Tensor
    matched_teacher_indices: Tensor


def teacher_obb_envelopes_xyxy(
    obbs_cxcywha: np.ndarray, *, coordinate_scale: float
) -> np.ndarray:
    """Convert pixel OBBs to axis-aligned envelopes without clipping."""

    obbs = np.asarray(obbs_cxcywha, dtype=np.float64)
    if obbs.ndim != 2 or obbs.shape[1] != 5:
        raise ValueError("teacher OBBs must have shape [N,5]")
    if (
        not np.isfinite(obbs).all()
        or np.any(obbs[:, 2:4] <= 0.0)
        or not np.isfinite(coordinate_scale)
        or coordinate_scale <= 0.0
    ):
        raise ValueError(
            "teacher OBBs and coordinate scale must be finite and positive"
        )
    scaled = obbs.copy()
    scaled[:, :4] *= float(coordinate_scale)
    cosine = np.abs(np.cos(scaled[:, 4]))
    sine = np.abs(np.sin(scaled[:, 4]))
    half_width = 0.5 * (cosine * scaled[:, 2] + sine * scaled[:, 3])
    half_height = 0.5 * (sine * scaled[:, 2] + cosine * scaled[:, 3])
    return np.stack(
        (
            scaled[:, 0] - half_width,
            scaled[:, 1] - half_height,
            scaled[:, 0] + half_width,
            scaled[:, 1] + half_height,
        ),
        axis=1,
    ).astype(np.float32)


def _pairwise_iou_xyxy(first: Tensor, second: Tensor) -> Tensor:
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def assign_predictions_to_teachers(
    *,
    prediction_boxes_xyxy: Tensor,
    prediction_labels: Tensor,
    teacher_obbs: np.ndarray,
    teacher_class_ids: np.ndarray,
    teacher_coordinate_scale: float,
    class_id_mapping: Mapping[int, int],
    min_iou: float,
    max_center_distance_px: float,
) -> TeacherAssignment:
    """Hungarian partial assignment with explicit class, IoU and centre gates."""

    boxes = torch.as_tensor(prediction_boxes_xyxy, dtype=torch.float32)
    labels = torch.as_tensor(prediction_labels, dtype=torch.long, device=boxes.device)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != (len(boxes),):
        raise ValueError("prediction boxes/labels must have shapes [N,4]/[N]")
    if not torch.isfinite(boxes).all() or ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
        raise ValueError(
            "prediction boxes must be finite xyxy boxes with positive area"
        )
    if not 0.0 <= min_iou <= 1.0 or max_center_distance_px <= 0.0:
        raise ValueError("assignment gates are out of range")
    teacher_ids = np.asarray(teacher_class_ids)
    teacher_boxes_np = teacher_obb_envelopes_xyxy(
        teacher_obbs, coordinate_scale=teacher_coordinate_scale
    )
    if teacher_ids.ndim != 1 or teacher_ids.shape[0] != teacher_boxes_np.shape[0]:
        raise ValueError("teacher class IDs must have shape [M]")
    if not np.issubdtype(teacher_ids.dtype, np.integer):
        raise ValueError("teacher class IDs must be integers")
    mapped = []
    for label in labels.tolist():
        if label not in class_id_mapping:
            raise ValueError(f"prediction class {label} has no teacher class mapping")
        mapped.append(int(class_id_mapping[label]))

    count = len(boxes)
    teacher_indices = torch.full((count,), -1, dtype=torch.long, device=boxes.device)
    quality = torch.full(
        (count,), float("nan"), dtype=torch.float32, device=boxes.device
    )
    center_quality = quality.clone()
    if count == 0 or len(teacher_boxes_np) == 0:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return TeacherAssignment(teacher_indices, quality, center_quality, empty, empty)

    teacher_boxes = torch.from_numpy(teacher_boxes_np).to(boxes.device)
    ious = _pairwise_iou_xyxy(boxes, teacher_boxes)
    prediction_centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    teacher_centers = 0.5 * (teacher_boxes[:, :2] + teacher_boxes[:, 2:])
    center_distances = torch.linalg.vector_norm(
        prediction_centers[:, None] - teacher_centers[None], dim=-1
    )
    mapped_labels = torch.tensor(mapped, dtype=torch.long, device=boxes.device)
    teacher_labels = torch.from_numpy(teacher_ids.astype(np.int64, copy=False)).to(
        boxes.device
    )
    valid = (
        (mapped_labels[:, None] == teacher_labels[None])
        & (ious >= float(min_iou))
        & (center_distances <= float(max_center_distance_px))
    )
    pairs = partial_linear_assignment(1.0 - ious, valid, miss_cost=1.0, new_cost=1.0)
    if pairs:
        prediction_rows = torch.tensor(
            [pair[0] for pair in pairs], dtype=torch.long, device=boxes.device
        )
        teacher_rows = torch.tensor(
            [pair[1] for pair in pairs], dtype=torch.long, device=boxes.device
        )
        teacher_indices[prediction_rows] = teacher_rows
        quality[prediction_rows] = ious[prediction_rows, teacher_rows]
        center_quality[prediction_rows] = center_distances[
            prediction_rows, teacher_rows
        ]
    else:
        prediction_rows = torch.empty(0, dtype=torch.long, device=boxes.device)
        teacher_rows = torch.empty(0, dtype=torch.long, device=boxes.device)
    return TeacherAssignment(
        teacher_indices=teacher_indices,
        ious=quality,
        center_distances_px=center_quality,
        matched_prediction_indices=prediction_rows,
        matched_teacher_indices=teacher_rows,
    )


__all__ = [
    "PREDICTION_FEATURE_CACHE_SCHEMA",
    "TeacherAssignment",
    "assign_predictions_to_teachers",
    "teacher_obb_envelopes_xyxy",
]
