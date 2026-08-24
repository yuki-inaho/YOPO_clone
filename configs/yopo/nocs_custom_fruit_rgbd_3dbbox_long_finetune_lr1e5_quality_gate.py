"""Five-epoch gate for the LR=1e-5 single-factor candidate."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e5.py"]

# Preserve the parent 80-epoch scheduler horizon so this is its first five
# epochs, while evaluating the full validation split once at epoch five.
train_cfg = dict(max_epochs=5, val_interval=5)
