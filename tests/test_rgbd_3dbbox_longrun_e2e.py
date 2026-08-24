import json
import math
import re
from pathlib import Path


WORK_DIR = Path("work_dirs/rgbd3d_long_ft80_lr1e6")
ERROR_MARKERS = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "FloatingPointError",
    "CUDA out of memory",
)


def test_rgbd_3dbbox_longrun_e2e():
    summary = json.loads((WORK_DIR / "longrun_summary.json").read_text())
    logs = sorted(WORK_DIR.glob("*/*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(errors="replace")

    train = re.findall(r"Epoch\(train\)\s*\[(\d+)\]\[\s*(\d+)/12\]", text)
    metrics = re.findall(
        r"Epoch\(val\) \[(\d+)\]\[50/50\].*?AP50: ([0-9.]+).*?"
        r"3d_iou_0.50: ([0-9.]+)",
        text,
    )
    assert train[-1] == ("80", "12")
    assert [int(row[0]) for row in metrics] == list(range(5, 81, 5))
    assert not any(marker in text for marker in ERROR_MARKERS)
    assert all(math.isfinite(float(value)) for row in metrics for value in row[1:])

    best = max(metrics, key=lambda row: float(row[2]))
    final = metrics[-1]
    assert int(best[0]) == summary["best_epoch"]
    assert float(best[2]) == summary["best_3d_iou_0.50"]
    assert float(final[1]) == summary["final_ap50"]
    assert float(final[2]) == summary["final_3d_iou_0.50"]
    assert len(metrics) == summary["validation_count"]
    assert summary["best_3d_iou_0.50"] >= summary["baseline_3d_iou_0.50"]
    assert summary["error_marker_count"] == 0
    assert summary["max_epochs"] == 80
    assert (WORK_DIR / summary["checkpoint"]).is_file()
