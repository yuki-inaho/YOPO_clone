"""Two-update capacity probe for the YOLO26n RGB-D model."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26n_stage1_full.py"]

train_dataloader = dict(
    batch_size=24,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
)
train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=2,
    val_interval=100,
)
val_dataloader = None
val_cfg = None
val_evaluator = None
custom_hooks = [dict(type="ScheduleFreeOptimizerModeHook")]
default_hooks = dict(checkpoint=None, logger=dict(interval=1))

work_dir = "work_dirs/yopo_yolo26n_rgbd_stage1_capacity"
