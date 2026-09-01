"""The depth anchor on the resolution that is best for 2D.

The two DoDs had split: 800x557 gives the best rotated-NMS mAP50 (0.7312) while
1024x712 gives the best 3D, and the anchored 1024 run reached shared AP20 0.4149
with a 4.8 mm range error but scored 0.6976 in 2D.  The anchor is a property of
the depth path, not of the resolution, so it should transfer.

Same recipe as the 1024 anchored run that worked -- anchor on every query, L1 on
z because the anchor's error has a heavy tail and L2 lets that tail pull the
residual toward the mean -- started from the 800 baseline with its z output
zeroed, so it begins at "trust the sensor" like the others.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
        loss_z=dict(_delete_=True, type="L1PoseLoss", loss_weight=50.0),
        sensor_depth_scale=3.90524303e-3,
        sensor_depth_anchor=True,
    ),
)

max_epochs = 12
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=1)

load_from = None
