"""Evidence-based end-to-end contract for the recorded RGB-D 3D BBOX run."""

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval"
SCALARS_PATH = RUN_DIR / "20260824_143138/vis_data/scalars.json"


def _records():
    return [json.loads(line) for line in SCALARS_PATH.read_text().splitlines()]


def test_rgbd_3dbbox_e2e():
    """The real 20-epoch run has finite training and complete validation proof."""
    assert SCALARS_PATH.is_file()

    records = _records()
    train_records = [item for item in records if "loss" in item and "grad_norm" in item]
    val_records = [item for item in records if "3d_iou_0.50" in item]

    assert len(train_records) == 20 * 12
    assert [item["step"] for item in val_records] == list(range(1, 21))
    assert all(
        math.isfinite(float(item[key]))
        for item in train_records
        for key in ("loss", "grad_norm")
    )
    assert all(
        math.isfinite(float(item[key]))
        for item in val_records
        for key in ("AP50", "3d_iou_0.10", "3d_iou_0.25", "3d_iou_0.50", "3d_iou_0.75")
    )

    report = json.loads((RUN_DIR / "partial_transfer_report.json").read_text())
    assert report["loaded_key_count"] == 300
    assert report["unexpected_keys"] == []

    best = RUN_DIR / "best_3d_iou_0.50_epoch_14.pth"
    assert best.is_file()
    assert best.stat().st_size > 0
