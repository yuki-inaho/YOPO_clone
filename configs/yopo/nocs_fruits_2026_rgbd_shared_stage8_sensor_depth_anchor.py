"""Stage 8: make metric RGB-D depth the CoP z anchor and learn a residual."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage7_ellipsoid_projection.py']

model = dict(
    bbox_head=dict(
        # The packed depth channel is metres: LoadRawDepthImageWithValidMask
        # divides uint16 millimetres by 1000 and the preprocessor uses std=1.
        sensor_depth_scale=1.0,
        sensor_depth_window=2,
        sensor_depth_anchor=True,
        # Anchor denoising queries too.  With the zeroed residual initialization
        # this avoids a large absolute-depth warm-up loss; noisy centres still
        # use the valid-window/fallback policy in the head.
        sensor_depth_anchor_denoising=True,
    ),
)

optim_wrapper = dict(
    optimizer=dict(lr=5e-5, aux_lr=5e-5),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.cop_z_': dict(lr_mult=1.0),
        },
    ),
)

# Keep the production default at the measured stable VRAM boundary rather
# than inheriting the B1 development setting from the shared base config.
train_dataloader = dict(
    batch_size=24,
    num_workers=6,
    persistent_workers=False,
    pin_memory=True,
)

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
load_from = None
resume = False
