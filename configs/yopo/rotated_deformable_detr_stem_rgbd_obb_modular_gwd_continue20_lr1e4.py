"""Additional 20-epoch GWD stage with a fresh, higher-LR optimizer."""

_base_ = ["./rotated_deformable_detr_stem_rgbd_obb_modular_gwd_full.py"]

optim_wrapper = dict(
    optimizer=dict(
        muon_lr=1e-4,
        sf_lr=1e-5,
    ),
)

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16],
        gamma=0.1,
    ),
]

# The completed 15-epoch GWD best is supplied explicitly on the CLI.
load_from = None
resume = False
