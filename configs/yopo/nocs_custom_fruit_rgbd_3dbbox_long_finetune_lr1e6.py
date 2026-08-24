"""Conservative single-factor continuation after the LR=1e-5 gate regressed."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune.py"]

# Keep every long-run contract unchanged.  The only mutation is a tenfold
# lower fresh-optimizer LR to test whether the epoch-14 optimum can be
# preserved before committing to a long continuation.
optim_wrapper = dict(optimizer=dict(lr=1e-6))
