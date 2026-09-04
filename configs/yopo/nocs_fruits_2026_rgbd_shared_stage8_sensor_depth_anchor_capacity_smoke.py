"""Two-update B24 capacity smoke for the metric-depth-anchor stage."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py']

load_from = None
resume = False
train_dataloader = dict(
    batch_size=24, num_workers=0, persistent_workers=False, pin_memory=True)
train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=2, val_interval=100)
val_dataloader = None
val_cfg = None
val_evaluator = None
custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]
default_hooks = dict(checkpoint=None, logger=dict(interval=1))
