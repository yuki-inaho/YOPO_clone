"""800x600 Stage B keeping the KLD centre term -- the control.

Identical to ``train_800_fix_center_off`` in every other respect, including the
initialization, so the difference between the two runs is attributable to the
single flag and to nothing else.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=True),
    ),
)
