"""Final evidence audit for the throughput-refactor and long RGB-D run."""

import hashlib
import json
import math
from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "work_dirs/rgbd3d_throughput_benchmark_v3"
LONG_DIR = REPO_ROOT / "work_dirs/rgbd3d_long_ft80_lr1e6"
WORKDOC = (
    REPO_ROOT
    / "temp/workdoc_Aug24-2026_rgbd3dbbox_throughput_refactor_longtrain.md"
)
LONG_CONFIG = (
    REPO_ROOT
    / "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py"
)


def test_benchmark_artifact_has_reproducible_snapshot_and_finite_timings():
    report = json.loads((BENCHMARK_DIR / "throughput_benchmark.json").read_text())
    snapshot = BENCHMARK_DIR / Path(report["provenance"]["config"]).name

    assert snapshot.is_file()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
        report["provenance"]["config_sha256"]
    )
    assert report["provenance"]["seed"] == 3407
    assert report["provenance"]["batch_size"] == 26
    assert report["warmup_iterations"] == 2
    assert report["train"]["measured_iterations"] == 10
    assert report["val"]["measured_iterations"] == 50
    assert report["cuda_max_memory_mib"] >= 19000
    for phase in ("train", "val"):
        for category in ("compute_seconds", "data_seconds"):
            assert all(
                math.isfinite(report[phase][category][stat])
                and report[phase][category][stat] >= 0
                for stat in ("min", "median", "p95", "max")
            )


def test_long_run_artifacts_and_config_provenance_are_consistent():
    summary = json.loads((LONG_DIR / "longrun_summary.json").read_text())
    config = Config.fromfile(LONG_CONFIG)

    assert summary["config"] == str(LONG_CONFIG.relative_to(REPO_ROOT))
    assert summary["max_epochs"] == config.train_cfg.max_epochs == 80
    assert config.train_cfg.val_interval == 5
    assert config.train_dataloader.batch_size == 26
    assert config.model.backbone.freeze_rgb is True
    assert config.optim_wrapper.type == "AmpScheduleFreeOptimWrapper"
    assert config.val_evaluator.type == "NOCSMetric"
    assert config.default_hooks.checkpoint.save_best == "3d_iou_0.50"
    assert config.default_hooks.checkpoint.save_last is False
    assert config.default_hooks.checkpoint.save_optimizer is False

    assert summary["validation_count"] == 16
    assert summary["validation_image_count"] == 50
    assert summary["error_marker_count"] == 0
    assert summary["best_epoch"] == 10
    assert summary["best_3d_iou_0.50"] >= summary["baseline_3d_iou_0.50"]
    assert summary["final_3d_iou_0.50"] < summary["best_3d_iou_0.50"]
    assert all(
        math.isfinite(summary[key])
        for key in (
            "baseline_3d_iou_0.50",
            "best_3d_iou_0.50",
            "final_3d_iou_0.50",
            "final_ap50",
            "wall_time_seconds",
            "peak_memory_mib",
        )
    )
    assert (LONG_DIR / summary["checkpoint"]).is_file()


def test_overlay_and_workdoc_trace_evidence_are_present():
    overlay_dir = LONG_DIR / "epoch10_overlays"
    manifest = json.loads((overlay_dir / "manifest.json").read_text())
    comparison = json.loads((overlay_dir / "comparison_manifest.json").read_text())
    workdoc = WORKDOC.read_text()

    assert manifest["selection_rule"] == "top-1 query per image, no confidence threshold"
    assert len(manifest["images"]) == 3
    assert all(Path(item["overlay_image"]).is_file() for item in manifest["images"])
    assert comparison["long_run"]["value"] > comparison["baseline"]["value"]
    assert "not practically accurate" in comparison["visual_review"]
    assert all(f"TR-{index}" in workdoc for index in range(1, 6))
    assert "実用3D detection精度は明確に未達" in workdoc
    # The audit test cannot require its own checkbox to be complete.  It does
    # require every preceding atomic checklist item to be complete; the final
    # command-level audit checks the whole document after DoD is recorded.
    before_this_test = workdoc.split(
        "🧪 **テスト**: `rgbd_3dbbox_throughput_longrun_audit`",
        maxsplit=1,
    )[0]
    assert "- [ ]" not in before_this_test
