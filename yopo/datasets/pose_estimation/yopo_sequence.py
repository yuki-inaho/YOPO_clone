"""Canonical variable-length RGB-D sequence data for YOPO context training."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from yopo.registry import DATASETS

_SPLITS = {"train", "val", "smoke"}
_ANNOTATION_KINDS = {"pseudo", "authoritative"}


class SequenceContractError(ValueError):
    """Raised when a manifest, frame, or window violates the sequence contract."""


@dataclass(frozen=True)
class FrameRecord:
    """Paths and camera metadata for one immutable frame artifact."""

    scene: str
    frame_id: int
    source_stem: str
    split: str
    color_path: Path
    depth_path: Path
    label_path: Path
    annotation_kind: str
    intrinsic: np.ndarray
    extrinsic_w2c: np.ndarray


@dataclass(frozen=True)
class WindowRecord:
    """One source-authored temporal row; it is never padded or synthesized."""

    sequence_id: str
    scene: str
    source_row: int
    split: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class WindowSample:
    """Decoded variable-length sequence consumed by the dedicated context runner."""

    split: str
    sequence_id: str
    frame_ids: tuple[int, ...]
    frames: tuple[dict[str, Any], ...]
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray
    valid_frame_mask: np.ndarray
    time_delta: np.ndarray
    annotation_kind: str
    augmentation: dict[str, bool]


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SequenceContractError(
            f"cannot read sequence manifest {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise SequenceContractError("sequence manifest must be a JSON object")
    if (
        document.get("schema") != "yopo_rgbd_sequence_manifest_v1"
        or document.get("schema_version") != 1
    ):
        raise SequenceContractError("unsupported YOPO sequence manifest schema")
    return document


def _resolve_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise SequenceContractError(f"invalid relative {field} path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise SequenceContractError(f"{field} path escapes manifest root: {value}")
    return path


class SequenceIndex:
    """Validate and index only the frame/window relationships in a manifest."""

    def __init__(self, manifest_path: str | Path, *, split: str) -> None:
        if split not in _SPLITS:
            raise SequenceContractError(f"unsupported split: {split}")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        document = _read_manifest(self.manifest_path)
        contract = document.get("frame_contract")
        if not isinstance(contract, Mapping):
            raise SequenceContractError("manifest frame_contract is missing")
        required_contract = {
            "rgb_dtype": "uint8",
            "rgb_channels": 3,
            "depth_dtype": "uint16",
            "depth_unit": "mm",
            "depth_invalid_value": 0,
        }
        for field, expected in required_contract.items():
            if contract.get(field) != expected:
                raise SequenceContractError(
                    f"frame contract {field}={contract.get(field)!r}, expected {expected!r}"
                )
        width = contract.get("width")
        height = contract.get("height")
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or min(width, height) <= 0
        ):
            raise SequenceContractError("frame width/height must be positive integers")
        self.frame_contract = dict(contract)
        annotation = document.get("annotation")
        if (
            not isinstance(annotation, Mapping)
            or annotation.get("kind") not in _ANNOTATION_KINDS
        ):
            raise SequenceContractError(
                "manifest annotation provenance is missing or invalid"
            )
        self.annotation_kind = str(annotation["kind"])

        raw_frames = document.get("frames")
        if not isinstance(raw_frames, list):
            raise SequenceContractError("manifest frames must be a list")
        records: dict[tuple[str, int], FrameRecord] = {}
        by_id: dict[int, list[FrameRecord]] = {}
        for raw_frame in raw_frames:
            if not isinstance(raw_frame, Mapping):
                raise SequenceContractError("manifest frame must be an object")
            scene = raw_frame.get("scene")
            frame_id = raw_frame.get("frame_id")
            frame_split = raw_frame.get("split")
            source_stem = raw_frame.get("source_stem")
            if not isinstance(scene, str) or not isinstance(frame_id, int):
                raise SequenceContractError("manifest frame has invalid scene/frame_id")
            if frame_split not in _SPLITS or not isinstance(source_stem, str):
                raise SequenceContractError(
                    f"manifest frame {scene}/{frame_id} has invalid metadata"
                )
            if raw_frame.get("annotation_kind") != self.annotation_kind:
                raise SequenceContractError(
                    f"manifest frame {scene}/{frame_id} provenance differs from manifest"
                )
            intrinsic = np.asarray(raw_frame.get("intrinsic"), dtype=np.float32)
            extrinsic = np.asarray(raw_frame.get("extrinsic_w2c"), dtype=np.float32)
            if intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4):
                raise SequenceContractError(
                    f"invalid camera shape for {scene}/{frame_id}"
                )
            if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
                raise SequenceContractError(f"non-finite camera for {scene}/{frame_id}")
            key = (scene, frame_id)
            if key in records:
                raise SequenceContractError(f"duplicate manifest frame: {key}")
            record = FrameRecord(
                scene=scene,
                frame_id=frame_id,
                source_stem=source_stem,
                split=str(frame_split),
                color_path=_resolve_path(self.root, raw_frame.get("color"), "color"),
                depth_path=_resolve_path(self.root, raw_frame.get("depth"), "depth"),
                label_path=_resolve_path(self.root, raw_frame.get("label"), "label"),
                annotation_kind=self.annotation_kind,
                intrinsic=intrinsic,
                extrinsic_w2c=extrinsic,
            )
            records[key] = record
            by_id.setdefault(frame_id, []).append(record)

        raw_windows = document.get("windows")
        if not isinstance(raw_windows, list):
            raise SequenceContractError("manifest windows must be a list")
        windows: list[WindowRecord] = []
        sequence_ids: set[str] = set()
        for raw_window in raw_windows:
            if not isinstance(raw_window, Mapping):
                raise SequenceContractError("manifest window must be an object")
            sequence_id = raw_window.get("sequence_id")
            scene = raw_window.get("scene")
            window_split = raw_window.get("split")
            source_row = raw_window.get("source_row")
            frame_ids = raw_window.get("frame_ids")
            if not isinstance(sequence_id, str) or sequence_id in sequence_ids:
                raise SequenceContractError(
                    f"invalid/duplicate sequence_id: {sequence_id!r}"
                )
            sequence_ids.add(sequence_id)
            if not isinstance(scene, str) or not isinstance(source_row, int):
                raise SequenceContractError(
                    f"window {sequence_id} has invalid scene/source row"
                )
            if (
                window_split not in _SPLITS
                or not isinstance(frame_ids, list)
                or not frame_ids
            ):
                raise SequenceContractError(
                    f"window {sequence_id} has invalid split/frame_ids"
                )
            if raw_window.get("length") != len(frame_ids):
                raise SequenceContractError(
                    f"window {sequence_id} length differs from frame_ids"
                )
            if any(not isinstance(value, int) for value in frame_ids):
                raise SequenceContractError(
                    f"window {sequence_id} frame IDs must be integers"
                )
            if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
                raise SequenceContractError(f"window {sequence_id} crosses a guard gap")
            for frame_id in frame_ids:
                record = records.get((scene, frame_id))
                if record is None:
                    raise SequenceContractError(
                        f"window {sequence_id} references a missing frame or guard frame"
                    )
                if record.split != window_split:
                    raise SequenceContractError(
                        f"window {sequence_id} crosses split boundaries"
                    )
            if window_split == split:
                windows.append(
                    WindowRecord(
                        sequence_id=sequence_id,
                        scene=scene,
                        source_row=source_row,
                        split=str(window_split),
                        frame_ids=tuple(frame_ids),
                    )
                )
        if not windows:
            raise SequenceContractError(f"manifest split has no windows: {split}")
        self.split = split
        self._records = records
        self._records_by_id = by_id
        self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> WindowRecord:
        return self.windows[index]

    def record(self, scene: str, frame_id: int) -> FrameRecord:
        return self._records[(scene, frame_id)]

    def frame(self, frame_id: int) -> FrameRecord:
        """Return a frame by ID only when it is unambiguous across scenes."""

        records = self._records_by_id.get(frame_id, [])
        if len(records) != 1:
            raise SequenceContractError(f"frame_id {frame_id} is missing or ambiguous")
        return records[0]


def _decode_rgbd_frame(
    record: FrameRecord, *, width: int, height: int
) -> dict[str, Any]:
    """Decode one synchronized RGB-D pair without touching its annotation."""

    for field, path in (("color", record.color_path), ("depth", record.depth_path)):
        if not path.is_file():
            raise SequenceContractError(f"missing {field} artifact: {path}")
    image = cv2.imread(str(record.color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(record.depth_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
        raise SequenceContractError(
            f"RGB must be uint8[{height},{width},3]: {record.color_path}"
        )
    if depth is None or depth.shape != (height, width) or depth.dtype != np.uint16:
        raise SequenceContractError(
            f"depth must be uint16[{height},{width}]: {record.depth_path}"
        )
    return {
        "scene": record.scene,
        "frame_id": record.frame_id,
        "source_stem": record.source_stem,
        "image_bgr": image,
        "depth_mm": depth,
        "depth_valid_mask": depth > 0,
        "intrinsic": record.intrinsic.copy(),
        "extrinsic_w2c": record.extrinsic_w2c.copy(),
    }


class RGBDFrameReader:
    """Decode inference RGB-D inputs while never opening annotation files."""

    def __init__(self, frame_contract: Mapping[str, Any]) -> None:
        self.width = int(frame_contract["width"])
        self.height = int(frame_contract["height"])

    def read(self, record: FrameRecord) -> dict[str, Any]:
        return _decode_rgbd_frame(record, width=self.width, height=self.height)


class FrameRecordReader:
    """Decode and validate one RGB/depth/YOPO-label triplet."""

    _ARRAY_SHAPES = {
        "class_ids": (),
        "instance_ids": (),
        "bboxes": (4,),
        "translations": (3,),
        "rotations": (3, 3),
        "sizes": (3,),
        "scales": (),
        "obb_cxcywha_rad": (5,),
        "dota_corners_xy": (4, 2),
        "ellipsoid_centers_cfov_m": (3,),
        "ellipsoid_axes_cfov": (3, 3),
        "ellipsoid_radii_m": (3,),
        "ellipsoid_valid": (),
    }

    def __init__(self, frame_contract: Mapping[str, Any]) -> None:
        self.width = int(frame_contract["width"])
        self.height = int(frame_contract["height"])

    def read(self, record: FrameRecord) -> dict[str, Any]:
        decoded = _decode_rgbd_frame(record, width=self.width, height=self.height)
        if not record.label_path.is_file():
            raise SequenceContractError(f"missing label artifact: {record.label_path}")
        try:
            with record.label_path.open("rb") as stream:
                label = pickle.load(stream)
        except (
            OSError,
            pickle.PickleError,
            EOFError,
            AttributeError,
            ValueError,
        ) as error:
            raise SequenceContractError(
                f"cannot read label {record.label_path}: {error}"
            ) from error
        if not isinstance(label, dict):
            raise SequenceContractError(
                f"label must be a dictionary: {record.label_path}"
            )
        class_ids = np.asarray(label.get("class_ids"))
        if class_ids.ndim != 1 or not np.issubdtype(class_ids.dtype, np.integer):
            raise SequenceContractError(f"invalid class_ids in {record.label_path}")
        count = len(class_ids)
        copied_label = dict(label)
        for field, tail in self._ARRAY_SHAPES.items():
            value = np.asarray(label.get(field))
            expected_shape = (count, *tail)
            if value.shape != expected_shape:
                raise SequenceContractError(
                    f"{field} shape {value.shape}, expected {expected_shape}: {record.label_path}"
                )
            if field != "ellipsoid_valid" and not np.issubdtype(value.dtype, np.number):
                raise SequenceContractError(
                    f"{field} must be numeric: {record.label_path}"
                )
            if (
                field != "ellipsoid_valid"
                and not np.isfinite(value.astype(np.float64)).all()
            ):
                raise SequenceContractError(
                    f"{field} is non-finite: {record.label_path}"
                )
            copied_label[field] = value.copy()
        source = label.get("source_metadata")
        if not isinstance(source, Mapping):
            raise SequenceContractError(
                f"label has no source_metadata: {record.label_path}"
            )
        if source.get("frame_id") != record.frame_id:
            raise SequenceContractError(f"label frame ID mismatch: {record.label_path}")
        if source.get("annotation_kind") != record.annotation_kind:
            raise SequenceContractError(
                f"label provenance mismatch: {record.label_path}"
            )
        return {
            **decoded,
            "annotation_kind": record.annotation_kind,
            "label": copied_label,
        }


def _normalize_pi_periodic(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle + np.pi / 2.0, np.pi) - np.pi / 2.0


def _reflect_intrinsic(value: Any, reflection: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float32)
    if intrinsic.shape == (4,):
        result = intrinsic.copy()
        result[0] *= -1.0
        result[2] = reflection[0, 2] - result[2]
        return result
    if intrinsic.shape == (9,):
        return (reflection @ intrinsic.reshape(3, 3)).reshape(-1)
    if intrinsic.shape == (3, 3):
        return reflection @ intrinsic
    raise SequenceContractError(
        f"unsupported intrinsic shape during flip: {intrinsic.shape}"
    )


def _horizontal_flip(frame: dict[str, Any], width: int) -> dict[str, Any]:
    flipped = dict(frame)
    flipped["image_bgr"] = np.flip(frame["image_bgr"], axis=1).copy()
    flipped["depth_mm"] = np.flip(frame["depth_mm"], axis=1).copy()
    flipped["depth_valid_mask"] = np.flip(frame["depth_valid_mask"], axis=1).copy()
    reflection = np.asarray(
        [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    flipped["intrinsic"] = reflection @ np.asarray(frame["intrinsic"], dtype=np.float32)
    label = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in frame["label"].items()
    }
    boxes = np.asarray(label["bboxes"]).copy()
    if len(boxes):
        old_x1 = boxes[:, 1].copy()
        old_x2 = boxes[:, 3].copy()
        boxes[:, 1] = width - old_x2
        boxes[:, 3] = width - old_x1
    label["bboxes"] = boxes
    obbs = np.asarray(label["obb_cxcywha_rad"]).copy()
    if len(obbs):
        obbs[:, 0] = width - 1 - obbs[:, 0]
        obbs[:, 4] = _normalize_pi_periodic(-obbs[:, 4])
    label["obb_cxcywha_rad"] = obbs
    corners = np.asarray(label["dota_corners_xy"]).copy()
    if len(corners):
        valid = np.any(corners != 0, axis=(1, 2))
        corners[valid, :, 0] = width - 1 - corners[valid, :, 0]
    label["dota_corners_xy"] = corners
    if "center_2d" in label:
        centers = np.asarray(label["center_2d"]).copy()
        centers[:, 0] = width - 1 - centers[:, 0]
        label["center_2d"] = centers
    if "intrinsic" in label:
        label["intrinsic"] = _reflect_intrinsic(label["intrinsic"], reflection)
    flipped["label"] = label
    return flipped


@DATASETS.register_module()
class YOPOSequenceDataset:
    """Return source-authored windows with one shared image-space augmentation."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        augmentation_policy: str = "identity",
        flip_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        if augmentation_policy not in {"identity", "horizontal_flip"}:
            raise SequenceContractError(
                f"unsupported sequence augmentation policy: {augmentation_policy}"
            )
        if not 0.0 <= flip_probability <= 1.0:
            raise SequenceContractError("flip_probability must be in [0, 1]")
        if augmentation_policy == "identity" and flip_probability != 0.0:
            raise SequenceContractError(
                "identity augmentation requires flip_probability=0"
            )
        self.index = SequenceIndex(manifest_path, split=split)
        self.reader = FrameRecordReader(self.index.frame_contract)
        self.augmentation_policy = augmentation_policy
        self.flip_probability = float(flip_probability)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> WindowSample:
        window = self.index[index]
        frames = tuple(
            self.reader.read(self.index.record(window.scene, frame_id))
            for frame_id in window.frame_ids
        )
        flip = False
        if self.augmentation_policy == "horizontal_flip":
            flip = bool(
                np.random.default_rng(self.seed + index * 104729).random()
                < self.flip_probability
            )
        if flip:
            width = int(self.index.frame_contract["width"])
            frames = tuple(_horizontal_flip(frame, width) for frame in frames)
        intrinsics = np.stack([frame["intrinsic"] for frame in frames]).astype(
            np.float32
        )
        extrinsics = np.stack([frame["extrinsic_w2c"] for frame in frames]).astype(
            np.float32
        )
        frame_ids = tuple(frame["frame_id"] for frame in frames)
        return WindowSample(
            split=window.split,
            sequence_id=window.sequence_id,
            frame_ids=frame_ids,
            frames=frames,
            intrinsics=intrinsics,
            extrinsics_w2c=extrinsics,
            valid_frame_mask=np.ones(len(frames), dtype=np.bool_),
            time_delta=(np.asarray(frame_ids, dtype=np.float32) - float(frame_ids[0])),
            annotation_kind=self.index.annotation_kind,
            augmentation={"horizontal_flip": flip},
        )


def collate_window_samples(samples: Sequence[WindowSample]) -> tuple[WindowSample, ...]:
    """Preserve variable lengths; padding is owned by the context model, not data loading."""

    if not samples:
        raise SequenceContractError("cannot collate an empty sequence batch")
    return tuple(samples)
