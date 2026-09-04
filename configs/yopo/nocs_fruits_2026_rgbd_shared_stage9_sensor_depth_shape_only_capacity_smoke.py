"""Two-update B24 capacity smoke for the stage-9 shape-only follow-up."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage9_sensor_depth_shape_only.py']

# Capacity must exercise the actual stage-9 parent rather than random model
# initialization; the projection loss is intentionally fail-closed on that
# untrained geometry.
load_from = (
    'work_dirs/yopo_shared_stage8_sensor_depth_anchor_b24_20260904/'
    'best_projection_shared_AP_25_epoch_5.pth'
)
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
