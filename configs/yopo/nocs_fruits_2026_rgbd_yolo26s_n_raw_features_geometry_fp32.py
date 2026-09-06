"""D: isolate FP32 geometry from the accepted RGB=s/depth=n checkpoint."""

_base_ = [
    "./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py"
]

model = dict(bbox_head=dict(geometry_float32=True))

load_from = (
    "work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/"
    "best_ellipsoid_shared_AP_25_epoch_20.pth"
)
resume = False
work_dir = "work_dirs/yopo_geometry_fp32_20260906"

max_epochs = 15
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="projection/shared_AP_25",
        rule="greater",
        min_delta=0.001,
        patience=2,
        strict=True,
        check_finite=True,
    ),
]
