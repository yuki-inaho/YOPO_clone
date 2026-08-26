"""Two-iteration GPU contract check for the dense-proposal RT-DETR."""

_base_ = ['./rotated_rt_detr_stem_rgbd_obb_hybrid_riou_full.py']

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(indices=8),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(indices=4),
)
test_dataloader = val_dataloader
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2,
    val_interval=3,
)
param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=2)
default_hooks = dict(logger=dict(interval=1))

