"""Eight actual training updates to measure the asymmetric feature model."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_full.py"]

train_dataloader = dict(batch_size=20, num_workers=0, persistent_workers=False)
train_cfg = dict(
    _delete_=True, type="IterBasedTrainLoop", max_iters=8, val_interval=100
)
val_dataloader = None
val_cfg = None
val_evaluator = None
custom_hooks = [dict(type="ScheduleFreeOptimizerModeHook")]
default_hooks = dict(checkpoint=None, logger=dict(interval=1))
work_dir = "work_dirs/yopo_yolo26s_n_raw_features_capacity"
