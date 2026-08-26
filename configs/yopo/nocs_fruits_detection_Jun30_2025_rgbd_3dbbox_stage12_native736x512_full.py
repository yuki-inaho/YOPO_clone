"""Final native-736x512 FULL run seeded by the adopted Stage-11 best.

Pass the selected model-only checkpoint through ``--cfg-options load_from=``;
do not use ``--resume`` across the input-geometry change.  The fixed native
input layer is loss-neutral, so a gated loss variant can override only its
loss key while reusing this config contract.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage12_native736x512_base.py'
]

load_from = None
resume = False

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

