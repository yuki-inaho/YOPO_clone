"""Evidence audit for the RGB-D 3D bounding-box transfer workdoc."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKDOC = ROOT / "temp/workdoc_Aug24-2026_rgbd_3dbbox_transfer.md"
WORK_DIR = ROOT / "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval"


def test_rgbd_3dbbox_workdoc_audit_has_completed_checklist_and_artifacts():
    """The completed workdoc and its final artifacts remain mutually consistent."""
    document = WORKDOC.read_text(encoding="utf-8")
    incomplete_checkboxes = re.findall(r"^\s*- \[ \] .+$", document, re.MULTILINE)
    assert incomplete_checkboxes == []

    report = json.loads((WORK_DIR / "partial_transfer_report.json").read_text())
    assert report["loaded_key_count"] == 300
    assert report["missing_target_rgb_keys"] == []
    assert report["unexpected_keys"] == []

    checkpoint = WORK_DIR / "best_3d_iou_0.50_epoch_14.pth"
    assert checkpoint.stat().st_size > 100_000_000

    prediction_dump = WORK_DIR / "epoch14_predictions.pkl"
    manifest = WORK_DIR / "epoch14_overlays/manifest.json"
    assert prediction_dump.stat().st_size > 1_000_000
    overlay_manifest = json.loads(manifest.read_text())
    assert overlay_manifest["selection_rule"] == "top-1 query per image, no confidence threshold"
    assert len(overlay_manifest["images"]) == 3
