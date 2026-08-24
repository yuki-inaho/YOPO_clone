"""Five-epoch gate for the LR=1e-6 conservative continuation."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py"]

train_cfg = dict(max_epochs=5, val_interval=5)
