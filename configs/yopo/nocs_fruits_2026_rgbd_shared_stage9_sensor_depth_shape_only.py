"""Stage 9: fine-tune GauCho shape without re-opening the centre escape route.

Stage 8 makes the CoP depth output a sensor-anchored residual.  Its remaining
3D error is dominated by ellipsoid extent/volume, while the 2D heads are
already useful.  This controlled follow-up changes only the direct GauCho KLD
from ``KL(target || prediction)`` with a centre term to shape-only supervision
and gives gradients to the ellipsoid branch; all other parameters keep their
stage-8 values through zero-LR parameter groups.
"""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py']

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(
            _delete_=True,
            type='Ellipsoid3DKLDLoss',
            loss_weight=2.0,
            tau=1.0,
            include_center=False,
            fail_on_invalid=True,
        ),
    ),
)

# A shape-only ablation: the shared RGB-D stack, detection/2D heads, CoP
# centre/depth, and projection target remain fixed.  Only the GauCho 3D
# shape branch is updated, so any 3D change is attributable to the hypothesis.
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-5, aux_lr=2.5e-5),
    paramwise_cfg=dict(
        custom_keys={
            '_delete_': True,
            'bbox_head.reg_ellipsoid_branch': dict(lr_mult=1.0),
            'backbone': dict(lr_mult=0.0),
            'bbox_head': dict(lr_mult=0.0),
            'backbone.rgb_backbone': dict(lr_mult=0.0),
            'backbone.depth_backbone': dict(lr_mult=0.0),
            'backbone.depth_adapters': dict(lr_mult=0.0),
            'backbone.depth_beta': dict(lr_mult=0.0),
            'neck': dict(lr_mult=0.0),
            'encoder': dict(lr_mult=0.0),
            'decoder': dict(lr_mult=0.0),
            'bbox_head.reg_branches': dict(lr_mult=0.0),
            'bbox_head.reg_centers_2d_branch': dict(lr_mult=0.0),
            'bbox_head.cop_': dict(lr_mult=0.0),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.0),
            'bbox_head.reg_ellipse2d_branch': dict(lr_mult=0.0),
        },
    ),
)

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
load_from = None
resume = False
