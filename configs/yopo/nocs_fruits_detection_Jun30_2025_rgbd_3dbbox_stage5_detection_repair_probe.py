"""Five-epoch gate before a long 2D-detection repair continuation.

The portable 3D curriculum learned usable metric pose geometry, but its 2D
AP50 remained close to zero.  This stage therefore gives the classification,
box-regression, neck, decoder, and query paths a 10x larger base learning rate
while keeping the pretrained RGB-D backbone and pose-specific branches at or
below the preceding stage's effective 1e-6 rate.  All 3D losses stay enabled so
the probe measures and constrains pose retention instead of silently detaching
the two tasks.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage3_full.py'
]

load_from = (
    'work_dirs/nocs_fruits_736x512_rgbd_3dbbox_stage4_plateau/'
    'best_3d_iou_0.50_epoch_50.pth'
)
resume = False
custom_hooks = []

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            # Feature extractors are already pretrained and remain effectively
            # frozen at 1e-7; the neck is intentionally updated at base LR.
            'backbone.rgb_backbone': dict(lr_mult=0.01),
            'backbone.depth_backbone': dict(lr_mult=0.01),
            'backbone.depth_adapters': dict(lr_mult=0.1),
            'backbone.depth_beta': dict(lr_mult=0.1),
            # These are the branches this repair stage is meant to adapt.
            'bbox_head.cls_branches': dict(lr_mult=1.0),
            'bbox_head.reg_branches': dict(lr_mult=1.0),
            # Preserve the converged 3D prediction path at effective 1e-6.
            'bbox_head.cop_': dict(lr_mult=0.1),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.1),
            'bbox_head.reg_z_branch': dict(lr_mult=0.1),
            'bbox_head.reg_size_branch': dict(lr_mult=0.1),
            'bbox_head.reg_rotation_branch': dict(lr_mult=0.1),
            # Encoder/decoder/query/neck use the base 1e-5 rate so that the
            # shared representation can actually repair 2D localization.
            'encoder': dict(lr_mult=1.0),
        },
    ),
)

probe_epochs = 5
train_cfg = dict(max_epochs=probe_epochs, val_interval=probe_epochs)
param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=probe_epochs,
        save_last=True,
        max_keep_ckpts=2,
        save_best=['AP50', '3d_iou_0.50'],
        rule=['greater', 'greater'],
        save_optimizer=False,
    ),
    logger=dict(interval=5),
)
