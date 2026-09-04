"""Two-update B24 smoke for the stage-10 reverse-KLD follow-up."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage10_reverse_kld.py']

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
