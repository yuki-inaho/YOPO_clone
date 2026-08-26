"""Optional box-only OCD strategy after the MAL probe passes its gate."""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage10_2d_anchor_mal_probe5.py'
]

# These ranges reproduce the legacy DINO box-noise magnitudes while making
# the strategy explicit, independently replaceable, min-size safe, and unable
# to perturb angle/depth/size/rotation.  Only this config key differs from the
# MAL probe; query count and positive:negative ratio stay unchanged.
model = dict(
    dn_cfg=dict(
        box_noise=dict(
            type='BoxOnlyOCDNoise',
            positive_noise_scale=0.5,
            negative_noise_scale=(0.5, 1.0),
            min_size=1e-4,
        ),
    ),
)

