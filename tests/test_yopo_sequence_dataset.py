"""Canonical variable-length YOPO RGB-D sequence dataset contracts."""

from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from mmengine.config import Config

from yopo.datasets.pose_estimation.yopo_sequence import (
    FrameRecordReader,
    RGBDFrameReader,
    SequenceContractError,
    SequenceIndex,
    YOPOSequenceDataset,
    collate_window_samples,
)
from yopo.registry import DATASETS


def _label(frame_id: int, intrinsic: np.ndarray) -> dict:
    return {
        "class_ids": np.asarray([1], dtype=np.int32),
        "instance_ids": np.asarray([1], dtype=np.int32),
        "bboxes": np.asarray([[1.0, 1.0, 5.0, 6.0]], dtype=np.float32),
        "translations": np.asarray([[0.1, 0.2, 1.0]], dtype=np.float32),
        "rotations": np.eye(3, dtype=np.float32)[None],
        "sizes": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        "scales": np.asarray([1.0], dtype=np.float32),
        "obb_cxcywha_rad": np.asarray([[3.0, 3.0, 4.0, 2.0, 0.2]], dtype=np.float32),
        "dota_corners_xy": np.asarray(
            [[[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]]],
            dtype=np.float32,
        ),
        "ellipsoid_centers_cfov_m": np.asarray([[0.1, 0.2, 1.0]], dtype=np.float32),
        "ellipsoid_axes_cfov": np.eye(3, dtype=np.float32)[None],
        "ellipsoid_radii_m": np.asarray([[0.05, 0.04, 0.03]], dtype=np.float32),
        "ellipsoid_valid": np.asarray([True]),
        "intrinsic": intrinsic,
        "image_size_wh": np.asarray([8, 6], dtype=np.int32),
        "source_metadata": {
            "frame_id": frame_id,
            "annotation_kind": "pseudo",
        },
    }


def _write_fixture(root: Path) -> Path:
    intrinsic = np.asarray(
        [[500.0, 0.0, 4.0], [0.0, 500.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    frames = []
    split_frames = {"train": (0, 1, 2, 3), "val": (5, 6, 7, 8)}
    for split, frame_ids in split_frames.items():
        directory = root / "real" / f"scene_000000_{split}_part000"
        directory.mkdir(parents=True)
        for frame_id in frame_ids:
            stem = f"{frame_id:04d}"
            color = directory / f"{stem}_color.png"
            depth = directory / f"{stem}_depth.png"
            label = directory / f"{stem}_label.pkl"
            image = np.repeat(np.arange(8, dtype=np.uint8)[None, :, None], 6, axis=0)
            image = np.repeat(image, 3, axis=2)
            cv2.imwrite(str(color), image)
            cv2.imwrite(
                str(depth),
                np.repeat(np.arange(1, 9, dtype=np.uint16)[None], 6, axis=0),
            )
            with label.open("wb") as stream:
                pickle.dump(_label(frame_id, intrinsic), stream)
            frames.append(
                {
                    "scene": "scene_000000",
                    "frame_id": frame_id,
                    "source_stem": f"frame_{frame_id:06d}",
                    "split": split,
                    "color": color.relative_to(root).as_posix(),
                    "depth": depth.relative_to(root).as_posix(),
                    "label": label.relative_to(root).as_posix(),
                    "annotation_kind": "pseudo",
                    "intrinsic": intrinsic.tolist(),
                    "extrinsic_w2c": np.eye(4, dtype=np.float32)[:3].tolist(),
                }
            )
    windows = [
        {
            "sequence_id": "scene_000000/train/source_row000000",
            "scene": "scene_000000",
            "source_row": 0,
            "split": "train",
            "length": 2,
            "frame_ids": [0, 1],
            "overlap": {},
        },
        {
            "sequence_id": "scene_000000/train/source_row000001",
            "scene": "scene_000000",
            "source_row": 1,
            "split": "train",
            "length": 3,
            "frame_ids": [1, 2, 3],
            "overlap": {},
        },
        {
            "sequence_id": "scene_000000/val/source_row000002",
            "scene": "scene_000000",
            "source_row": 2,
            "split": "val",
            "length": 4,
            "frame_ids": [5, 6, 7, 8],
            "overlap": {},
        },
    ]
    manifest = {
        "schema": "yopo_rgbd_sequence_manifest_v1",
        "schema_version": 1,
        "source_format": "colmap_rgbd_v1",
        "frame_contract": {
            "width": 8,
            "height": 6,
            "rgb_dtype": "uint8",
            "rgb_channels": 3,
            "depth_dtype": "uint16",
            "depth_unit": "mm",
            "depth_invalid_value": 0,
        },
        "annotation": {"kind": "pseudo"},
        "metadata_sha256": {"source/dataset.json": "a" * 64},
        "source_frame_count": 9,
        "guard_frame_count": 1,
        "splits": {
            "train": {"frame_count": 4, "window_count": 2},
            "val": {"frame_count": 4, "window_count": 1},
            "smoke": {"frame_count": 0, "window_count": 0},
        },
        "frames": frames,
        "windows": windows,
    }
    path = root / "sequence_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_sequence_index_preserves_variable_source_windows_without_padding(
    tmp_path: Path,
) -> None:
    manifest = _write_fixture(tmp_path)

    train = YOPOSequenceDataset(manifest_path=manifest, split="train")
    val = YOPOSequenceDataset(manifest_path=manifest, split="val")

    assert len(train) == 2
    assert [len(train[index].frames) for index in range(len(train))] == [2, 3]
    assert val[0].frame_ids == (5, 6, 7, 8)
    assert train[0].valid_frame_mask.tolist() == [True, True]
    assert train[1].valid_frame_mask.tolist() == [True, True, True]
    assert train[0].time_delta.tolist() == [0.0, 1.0]
    assert all(frame["annotation_kind"] == "pseudo" for frame in train[0].frames)


def test_sequence_index_rejects_cross_split_and_guard_frames(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cross_split = deepcopy(manifest)
    cross_split["windows"][0]["split"] = "val"
    manifest_path.write_text(json.dumps(cross_split), encoding="utf-8")
    with pytest.raises(SequenceContractError, match="crosses split"):
        SequenceIndex(manifest_path, split="train")

    guard = deepcopy(manifest)
    guard["windows"][0]["frame_ids"] = [3, 4]
    manifest_path.write_text(json.dumps(guard), encoding="utf-8")
    with pytest.raises(SequenceContractError, match="missing frame|guard"):
        SequenceIndex(manifest_path, split="train")


def test_window_horizontal_flip_is_shared_and_geometry_consistent(
    tmp_path: Path,
) -> None:
    manifest = _write_fixture(tmp_path)
    dataset = YOPOSequenceDataset(
        manifest_path=manifest,
        split="train",
        augmentation_policy="horizontal_flip",
        flip_probability=1.0,
        seed=7,
    )

    sample = dataset[0]

    assert sample.augmentation == {"horizontal_flip": True}
    for frame in sample.frames:
        np.testing.assert_array_equal(frame["image_bgr"][0, :, 0], np.arange(7, -1, -1))
        np.testing.assert_array_equal(frame["depth_mm"][0], np.arange(8, 0, -1))
        np.testing.assert_allclose(frame["intrinsic"][0], [-500.0, 0.0, 3.0])
        np.testing.assert_allclose(frame["label"]["bboxes"][0], [1.0, 2.0, 5.0, 7.0])
        np.testing.assert_allclose(
            frame["label"]["obb_cxcywha_rad"][0],
            [4.0, 3.0, 4.0, 2.0, -0.2],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(frame["label"]["translations"], [[0.1, 0.2, 1.0]])
        np.testing.assert_allclose(frame["extrinsic_w2c"], np.eye(4)[:3])


def test_reader_rejects_missing_path_bad_depth_and_nonfinite_camera(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture(tmp_path)
    index = SequenceIndex(manifest_path, split="train")
    record = index.frame(0)
    record.depth_path.unlink()
    with pytest.raises(SequenceContractError, match="missing depth"):
        FrameRecordReader(index.frame_contract).read(record)

    bad_depth_root = tmp_path / "bad_depth"
    bad_depth_manifest = _write_fixture(bad_depth_root)
    bad_index = SequenceIndex(bad_depth_manifest, split="train")
    bad_record = bad_index.frame(0)
    cv2.imwrite(str(bad_record.depth_path), np.ones((6, 8), dtype=np.uint8))
    with pytest.raises(SequenceContractError, match="uint16"):
        FrameRecordReader(bad_index.frame_contract).read(bad_record)

    camera_root = tmp_path / "camera"
    camera_manifest = _write_fixture(camera_root)
    document = json.loads(camera_manifest.read_text(encoding="utf-8"))
    document["frames"][0]["intrinsic"][0][0] = float("nan")
    camera_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SequenceContractError, match="non-finite camera"):
        SequenceIndex(camera_manifest, split="train")


def test_rgbd_inference_reader_never_requires_or_opens_label(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    index = SequenceIndex(manifest_path, split="train")
    record = index.frame(0)
    record.label_path.unlink()

    frame = RGBDFrameReader(index.frame_contract).read(record)

    assert set(frame) == {
        "scene",
        "frame_id",
        "source_stem",
        "image_bgr",
        "depth_mm",
        "depth_valid_mask",
        "intrinsic",
        "extrinsic_w2c",
    }
    assert frame["image_bgr"].shape == (6, 8, 3)
    assert frame["depth_mm"].dtype == np.uint16


def test_sequence_collate_keeps_variable_lengths_and_registry_config(
    tmp_path: Path,
) -> None:
    manifest = _write_fixture(tmp_path)
    dataset = DATASETS.build(
        dict(type="YOPOSequenceDataset", manifest_path=manifest, split="train")
    )

    batch = collate_window_samples([dataset[0], dataset[1]])
    config = Config.fromfile(
        "configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_sequence_context.py"
    )

    assert isinstance(dataset, YOPOSequenceDataset)
    assert [len(sample.frames) for sample in batch] == [2, 3]
    assert (
        config.sequence_context.manifest_path
        == "data/yopo_sequence_640x480/sequence_manifest.json"
    )
    assert config.sequence_context.initial_checkpoint.endswith("epoch_10.pth")
    assert config.sequence_context.checkpoint_kind == "raw"
    assert config.sequence_context.modes == ["B0", "B1", "B2", "C0", "C1"]
