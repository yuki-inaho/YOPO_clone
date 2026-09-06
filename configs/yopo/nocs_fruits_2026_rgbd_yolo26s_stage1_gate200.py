"""Finite validation/checkpoint/resume gate for YOLO26s RGB-D."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_stage1_full.py"]

train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=200,
    val_interval=100,
)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=100),
    logger=dict(interval=10),
)

work_dir = "work_dirs/yopo_yolo26s_rgbd_stage1_gate200"
