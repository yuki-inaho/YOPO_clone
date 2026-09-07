"""Label-free conversion from frame detector predictions to tracker inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

from yopo.structures import DetDataSample

from .geometry_context import camera_to_world_points
from .online_tracker import Detection
from .sequence_training import (
    CHECKPOINT_SCHEMA,
    make_identity_head,
    sample_pyramid_at_centers,
    sha256_file,
)


class IdentityDescriptor(Protocol):
    """Minimal descriptor boundary shared by trained and test heads."""

    def __call__(self, appearance: Tensor, positions_world: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class PredictionObservationBatch:
    """Raw prediction observations shared by cache extraction and inference."""

    prediction_indices: Tensor
    labels: Tensor
    bboxes_xyxy: Tensor
    scores: Tensor
    centers_px: Tensor
    centers_camera: Tensor
    centers_world: Tensor
    geometry_valid: Tensor
    appearance: Tensor


@dataclass(frozen=True)
class DetectionObservation:
    """Auditable detector-derived observation and its tracker projection."""

    prediction_index: int
    label: int
    bbox_xyxy: Tensor
    center_px: Tensor
    center_camera: Tensor | None
    appearance: Tensor
    detection: Detection


def load_sequence_descriptor(
    checkpoint_path: Path | None,
    *,
    requested_mode: str | None,
    allow_untrained: bool,
    appearance_dim: int,
    head_config: Any,
    detector_sha256: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load one descriptor with strict detector provenance and state parity."""

    if checkpoint_path is None:
        if not allow_untrained:
            raise ValueError(
                "--descriptor-checkpoint is required unless "
                "--allow-untrained-descriptor is explicit"
            )
        mode = requested_mode or "C1"
        torch.manual_seed(seed)
        descriptor = make_identity_head(
            mode=mode,
            appearance_dim=appearance_dim,
            head_config=head_config,
        ).to(device)
        return descriptor.eval(), {
            "kind": "random_untrained",
            "mode": mode,
            "checkpoint_sha256": None,
            "plumbing_only": True,
        }

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"descriptor checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported sequence descriptor checkpoint schema")
    mode = str(checkpoint.get("mode"))
    if requested_mode is not None and requested_mode != mode:
        raise ValueError("requested descriptor mode differs from checkpoint")
    if int(checkpoint.get("appearance_dim", -1)) != appearance_dim:
        raise ValueError("descriptor appearance dimension differs from detector")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("g10_checkpoint_sha256") != detector_sha256:
        raise ValueError("descriptor was not trained from this G10 checkpoint")
    descriptor = make_identity_head(
        mode=mode,
        appearance_dim=appearance_dim,
        head_config=checkpoint["head_config"],
    ).to(device)
    descriptor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return descriptor.eval(), {
        "kind": "trained_raw",
        "mode": mode,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "plumbing_only": False,
    }


def make_detector_data_sample(
    *,
    frame_id: int,
    image_path: str,
    intrinsic: Tensor | Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> DetDataSample:
    """Build the shared metadata contract for rescaled G10 prediction."""

    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("image_size must contain positive height and width")
    intrinsic_tensor = torch.as_tensor(intrinsic, dtype=torch.float32)
    if intrinsic_tensor.shape != (3, 3) or not torch.isfinite(intrinsic_tensor).all():
        raise ValueError("intrinsic must be a finite [3,3] matrix")
    sample = DetDataSample()
    sample.set_metainfo(
        {
            "img_id": int(frame_id),
            "img_path": str(image_path),
            "ori_shape": image_size,
            "img_shape": image_size,
            "pad_shape": image_size,
            "batch_input_shape": image_size,
            "scale_factor": (1.0, 1.0),
            "intrinsic": intrinsic_tensor.tolist(),
        }
    )
    return sample


def _require_prediction_tensor(
    predictions: object, field: str, shape_tail: tuple[int, ...]
) -> Tensor:
    value = getattr(predictions, field, None)
    if not isinstance(value, Tensor):
        raise ValueError(f"detector predictions require tensor field {field!r}")
    if value.ndim != 1 + len(shape_tail) or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(
            f"prediction field {field!r} must have shape [N,{','.join(map(str, shape_tail))}]"
        )
    return value


def _homogeneous_extrinsic(extrinsic_w2c: Tensor, reference: Tensor) -> Tensor:
    extrinsic = torch.as_tensor(extrinsic_w2c, device=reference.device).to(
        dtype=torch.float32
    )
    if extrinsic.shape == (3, 4):
        result = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)
        result[:3] = extrinsic
        return result
    if extrinsic.shape != (4, 4):
        raise ValueError("extrinsic_w2c must have shape [3,4] or [4,4]")
    return extrinsic


def build_detection_observations(
    predictions: object,
    backbone_features: Sequence[Tensor],
    descriptor: IdentityDescriptor,
    *,
    image_size: tuple[int, int],
    extrinsic_w2c: Tensor,
    score_threshold: float,
    max_detections: int,
    batch_index: int = 0,
) -> tuple[DetectionObservation, ...]:
    """Build causal observations using only detector outputs and current frame data."""

    batch = build_prediction_observation_batch(
        predictions,
        backbone_features,
        image_size=image_size,
        extrinsic_w2c=extrinsic_w2c,
        score_threshold=score_threshold,
        max_detections=max_detections,
        batch_index=batch_index,
    )
    if len(batch.prediction_indices) == 0:
        return ()
    embeddings = descriptor(batch.appearance, batch.centers_world)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(batch.prediction_indices):
        raise ValueError("descriptor must return shape [N,D]")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("descriptor returned nonfinite embeddings")

    observations = []
    for row, prediction_index in enumerate(batch.prediction_indices.tolist()):
        geometry_valid = bool(batch.geometry_valid[row])
        detection = Detection(
            center_world=batch.centers_world[row] if geometry_valid else None,
            embedding=embeddings[row],
            confidence=float(batch.scores[row]),
        )
        observations.append(
            DetectionObservation(
                prediction_index=prediction_index,
                label=int(batch.labels[row]),
                bbox_xyxy=batch.bboxes_xyxy[row],
                center_px=batch.centers_px[row],
                center_camera=batch.centers_camera[row] if geometry_valid else None,
                appearance=batch.appearance[row],
                detection=detection,
            )
        )
    return tuple(observations)


def build_prediction_observation_batch(
    predictions: object,
    backbone_features: Sequence[Tensor],
    *,
    image_size: tuple[int, int],
    extrinsic_w2c: Tensor,
    score_threshold: float,
    max_detections: int,
    batch_index: int = 0,
) -> PredictionObservationBatch:
    """Extract descriptor inputs from the exact causal prediction boundary."""

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0,1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    bboxes = _require_prediction_tensor(predictions, "bboxes", (4,))
    scores = _require_prediction_tensor(predictions, "scores", ())
    labels = _require_prediction_tensor(predictions, "labels", ())
    translations = _require_prediction_tensor(predictions, "translations", (3,))
    count = bboxes.shape[0]
    if (
        scores.shape[0] != count
        or labels.shape[0] != count
        or translations.shape[0] != count
    ):
        raise ValueError("detector prediction fields must have equal length")
    if not torch.isfinite(scores).all() or not torch.isfinite(bboxes).all():
        raise ValueError("detector scores and bboxes must be finite")
    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("image_size must contain positive height and width")
    valid_boxes = (
        (bboxes[:, 0] >= 0)
        & (bboxes[:, 1] >= 0)
        & (bboxes[:, 2] <= width)
        & (bboxes[:, 3] <= height)
        & (bboxes[:, 2] > bboxes[:, 0])
        & (bboxes[:, 3] > bboxes[:, 1])
    )
    if not valid_boxes.all():
        raise ValueError("detector bboxes must be nonempty and inside the image")
    retained = (scores >= score_threshold).nonzero(as_tuple=False).flatten()
    retained = retained[:max_detections]

    selected_boxes = bboxes[retained]
    centers_px = 0.5 * (selected_boxes[:, :2] + selected_boxes[:, 2:])
    appearance = sample_pyramid_at_centers(
        backbone_features,
        batch_index=batch_index,
        centers_px=centers_px,
        image_size=image_size,
    ).float()
    selected_camera = translations[retained].float()
    geometry_valid = torch.isfinite(selected_camera).all(dim=1)
    positions_world = torch.full_like(selected_camera, float("nan"))
    if geometry_valid.any():
        extrinsic = _homogeneous_extrinsic(extrinsic_w2c, selected_camera)
        positions_world[geometry_valid] = camera_to_world_points(
            selected_camera[geometry_valid], extrinsic
        )
    return PredictionObservationBatch(
        prediction_indices=retained,
        labels=labels[retained],
        bboxes_xyxy=selected_boxes,
        scores=scores[retained],
        centers_px=centers_px,
        centers_camera=selected_camera,
        centers_world=positions_world,
        geometry_valid=geometry_valid,
        appearance=appearance,
    )
