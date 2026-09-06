"""RGB=s, depth=n; transfer raw DEIM backbone and hybrid encoder per branch."""

_base_ = ["./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py"]

custom_imports = dict(
    imports=_base_.custom_imports.imports + ["yopo.models.backbones.yolo26_features"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        beta_init=0.1,
        rgb_backbone=dict(
            _delete_=True, type="YOLO26FeatureBackbone", scale="s", in_channels=3
        ),
        depth_backbone=dict(
            _delete_=True, type="YOLO26FeatureBackbone", scale="n", in_channels=1
        ),
    ),
    neck=dict(_delete_=True, type="EncodedPyramidNeck"),
    bbox_head=dict(cop_depth_context=dict(in_channels=[256, 256, 256])),
)

load_from = "work_dirs/yolo26s_n_raw_features_initial.pth"
resume = False
work_dir = "work_dirs/yopo_yolo26s_n_raw_features_full_20260906"
