"""Single-iteration RTX 5090 batch-capacity probe."""

_base_ = ['./rotated_rt_detr_stem_rgbd_obb_hybrid_riou_full.py']

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=False,
    dataset=dict(indices=32),
)
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=1,
    val_interval=2,
)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(logger=dict(interval=1))
