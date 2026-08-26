"""One-epoch native-736x512 KFIoU gate with corrected validation.

Run independently from the same selected Stage-11 model-only checkpoint used
by FULL.  The gate writes metrics and logs but no redundant model checkpoint.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_kfiou_base.py'
]

load_from = None
resume = False

train_cfg = dict(max_epochs=1, val_interval=1)

default_hooks = dict(
    checkpoint=None,
    logger=dict(interval=5),
)
