"""Two-update capacity smoke for the shared RGB-D direct ellipsoid stage."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage6_ellipsoid_direct.py']

load_from = None
resume = False
train_dataloader = dict(
    batch_size=16, num_workers=0, persistent_workers=False, pin_memory=True)
train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=2, val_interval=100)
val_dataloader = None
val_cfg = None
val_evaluator = None
custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]
default_hooks = dict(checkpoint=None, logger=dict(interval=1))
