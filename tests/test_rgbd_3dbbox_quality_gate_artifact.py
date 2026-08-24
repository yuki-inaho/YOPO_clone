import math
import re
from pathlib import Path

import pytest
from mmengine.config import Config


ERROR_MARKERS = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "FloatingPointError",
    "CUDA out of memory",
)


@pytest.mark.parametrize(
    ("work_dir", "config"),
    [
        (
            Path("work_dirs/rgbd3d_long_ft80_quality_gate"),
            "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune_quality_gate.py",
        ),
        (
            Path("work_dirs/rgbd3d_long_ft80_lr1e5_quality_gate"),
            "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e5_quality_gate.py",
        ),
        (
            Path("work_dirs/rgbd3d_long_ft80_lr1e6_quality_gate"),
            "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6_quality_gate.py",
        ),
    ],
)
def test_quality_gate_artifact_has_a_finite_full_validation_record(work_dir, config):
    cfg = Config.fromfile(config)
    assert cfg.train_cfg.max_epochs == 5
    assert cfg.train_cfg.val_interval == 5
    assert cfg.load_from.endswith("best_3d_iou_0.50_epoch_14.pth")

    logs = sorted(work_dir.glob("*/*.log"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert log_text.count("Epoch(train)") == 60
    assert not any(marker in log_text for marker in ERROR_MARKERS)

    final = re.search(
        r"Epoch\(val\) \[5\]\[50/50\].*?AP50: ([0-9.]+).*?"
        r"3d_iou_0.50: ([0-9.]+)",
        log_text,
    )
    assert final is not None
    ap50, iou50 = (float(value) for value in final.groups())
    assert math.isfinite(ap50)
    assert math.isfinite(iou50)
    assert 0.0 <= ap50 <= 1.0
    assert 0.0 <= iou50 <= 1.0
    assert (work_dir / "best_3d_iou_0.50_epoch_5.pth").is_file()
    assert (work_dir / "partial_transfer_report.json").is_file()
