"""One-epoch, small-slice smoke run for the RGB-D OBB contract."""

_base_ = ["./rotated_deformable_detr_stem_rgbd_obb_riou_stage1.py"]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(indices=64),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(indices=32),
)
test_dataloader = val_dataloader

max_epochs = 1
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=2)
