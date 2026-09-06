"""YOPO RGB-D FULL training with a transferred YOLO26s RGB backbone."""

_base_ = ["./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py"]

model = dict(
    backbone=dict(
        beta_init=0.0,
        rgb_backbone=dict(
            _delete_=True,
            type="YOLO26Backbone",
            scale="s",
            in_channels=3,
            return_idx=[1, 2, 3],
            init_cfg=None,
        ),
    ),
    neck=dict(in_channels=[256, 256, 512]),
)

# Weight-only Rotated-best backbone + stage-8 reusable YOPO initialization.
load_from = "work_dirs/yolo26s_rgbd_stage1_initial.pth"
resume = False

work_dir = "work_dirs/yopo_yolo26s_rgbd_stage1_full_20260906"
