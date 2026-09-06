"""Eight updates to measure the FP32-geometry B25 memory boundary."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_fp32.py"]

train_dataloader = dict(num_workers=0, persistent_workers=False)
train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=8,
    val_interval=100,
)
val_dataloader = None
val_cfg = None
val_evaluator = None
custom_hooks = [dict(type="ScheduleFreeOptimizerModeHook")]
default_hooks = dict(checkpoint=None, logger=dict(interval=1))
work_dir = "work_dirs/yopo_geometry_fp32_capacity_20260906"
