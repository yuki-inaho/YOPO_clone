"""YOPO RGB-D FULL training with a transferred YOLO26n RGB backbone."""

_base_ = ["./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py"]

model = dict(
    backbone=dict(
        beta_init=0.0,
        rgb_backbone=dict(
            _delete_=True,
            type="YOLO26Backbone",
            scale="n",
            in_channels=3,
            return_idx=[1, 2, 3],
            init_cfg=None,
        ),
    ),
    neck=dict(in_channels=[128, 128, 256]),
)

# Weight-only Rotated-best backbone + stage-8 reusable YOPO initialization.
load_from = "work_dirs/yolo26n_rgbd_stage1_initial.pth"
resume = False

work_dir = "work_dirs/yopo_yolo26n_rgbd_stage1_full_20260906"
