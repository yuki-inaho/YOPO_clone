"""Low-LR GWD continuation with validation-mAP plateau stopping.

This stage starts from the measured epoch-49 best of the preceding 50-epoch
run.  The checkpoint intentionally contains weights only, so the optimizer is
rebuilt at the already-decayed learning rates instead of restarting the former
high-LR phase.
"""

_base_ = ['./rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_full.py']

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Keep the converged-stage rates constant. ScheduleFree handles the vector
# parameters internally; an additional external LR schedule is unnecessary.
optim_wrapper = dict(
    optimizer=dict(
        muon_lr=1e-5,
        sf_lr=5e-7,
    ),
)
param_scheduler = []

# A gain below 0.0005 mAP50 is treated as noise. Twelve consecutive validation
# epochs without a larger gain is the operational definition of saturation.
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='rbbox_mAP_50',
        rule='greater',
        min_delta=5e-4,
        patience=12,
        strict=True,
        check_finite=True,
    ),
]

load_from = None
resume = False
