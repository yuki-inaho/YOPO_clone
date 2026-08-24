"""Five-epoch quality gate using the initial trajectory of the long fine-tune."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune.py"]

# Do not replace the inherited 80-epoch cosine schedule: this gate evaluates
# precisely the first five epochs that the actual long run will take.  It is a
# fresh process from the epoch-14 source, never a resume of this gate.
train_cfg = dict(max_epochs=5, val_interval=5)
