"""GWD continuation for dense RGB-D rotated RTMDet."""

_base_ = ['./rotated_rtmdet_stem_rgbd_obb_dense_riou_full.py']

model = dict(
    bbox_head=dict(
        loss_iou=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='gwd',
            fun='log1p',
            tau=1,
            loss_weight=2.0,
        ),
    ),
)

max_epochs = 15
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[12],
        gamma=0.1,
    ),
]
custom_hooks = []
load_from = None
resume = False

