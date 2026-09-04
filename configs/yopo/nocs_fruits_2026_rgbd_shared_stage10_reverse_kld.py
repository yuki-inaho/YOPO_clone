"""Stage 10: test reverse direct-ellipsoid KLD from the stage-8 best."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py']

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(direction='prediction_to_target'),
    ),
)

max_epochs = 15
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
