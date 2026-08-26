"""Selected native-736x512 objective: covariance KFIoU at probe weight 1.

This is the single loss/optimizer mixin shared by every runnable Stage-12
KFIoU config.  Native image geometry remains owned by the loss-neutral base;
artifact paths remain launch-time inputs.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_base.py'
]

load_from = None
resume = False

model = dict(
    bbox_head=dict(
        loss_obb_aux=dict(
            _delete_=True,
            type='GaussianKFIoULoss',
            loss_weight=1.0,
            fail_on_invalid=True,
            eps=1e-7,
        ),
    ),
)

# Keep the accepted mature-checkpoint probe LR.  Capacity selection between
# physical batches 15 and 16 does not introduce a second optimization change.
optim_wrapper = dict(
    optimizer=dict(lr=3e-6),
)
param_scheduler = []

# Runnable children add only the lifecycle hooks they actually need.
custom_hooks = [dict(type='ScheduleFreeOptimizerModeHook')]
