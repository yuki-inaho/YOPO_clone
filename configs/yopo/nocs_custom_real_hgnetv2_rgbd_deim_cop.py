_base_ = ['./nocs_custom_real_hgnetv2_rgbd_deim.py']

# CoP (Chain-of-Prediction) auxiliary-head fine-tune config.
#
# Reuses the fully-trained 100-epoch checkpoint (work_dirs/full_run/epoch_100.pth)
# via load_from (strict=False skips the newly added cop_* parameters which get
# random init), then trains only a bit more so the CoP chain (size -> rotation
# -> depth, MonoCoP arXiv:2505.04594) learns to exploit inter-attribute
# correlations. The plain parallel branches keep working unchanged.

# ── Enable the CoP auxiliary chain in the 9D pose head ────────────────────
model = dict(
    bbox_head=dict(use_cop_chain=True))

# Skipping resume: CoP adds new params, so we load the base weight only once.
resume = False

# Start from the converged 100-epoch checkpoint. strict=False (default for
# load_from) drops unexpected cop_* keys and keeps the pretrained weights for
# backbone / parallel head.
load_from = 'work_dirs/full_run/epoch_100.pth'

# A short continuation schedule: ~50 more epochs (~40 min) is plenty to let the
# CoP aux heads converge on top of the quality base features.
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True),
]
