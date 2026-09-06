"""F-on: exact shape-head-only control retaining the KLD centre term."""

_base_ = [
    "./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_fp32_aux.py"
]

custom_imports = dict(
    imports=_base_.custom_imports.imports + ["yopo.engine.hooks.freeze_except"],
    allow_failed_imports=False,
)
model = dict(bbox_head=dict(loss_ellipsoid=dict(include_center=True)))
custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="FreezeExceptHook",
        trainable_patterns=(r"^bbox_head\.reg_ellipsoid_branch\..+$",),
    ),
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
work_dir = "work_dirs/yopo_shape_center_on_20260906"
