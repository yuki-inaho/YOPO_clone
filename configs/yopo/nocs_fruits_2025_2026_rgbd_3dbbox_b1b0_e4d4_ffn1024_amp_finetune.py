"""AMP fine-tuning policy for the compact joint 2025+2026 RGB-D model.

The compact checkpoint is a weight-only transplant, not an optimizer resume.
Pass it as ``load_from``.  The base LR is deliberately higher than the mature
Stage-12 continuation LR because the B1 RGB path, depth adapters, and neck
must recover after topology surgery.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_3dbbox_"
    "b1b0_e4d4_ffn1024_native_800x600.py"
]

base_lr = 1e-4

train_dataloader = dict(batch_size=30)

# Delete all inherited stage-specific multipliers.  Keeping those accumulated
# keys would reduce parts of the backbone to 2e-8 and is inappropriate for a
# newly transplanted compact topology.
optim_wrapper = dict(
    _delete_=True,
    type="AmpScheduleFreeOptimWrapper",
    constructor="DefaultOptimWrapperConstructor",
    # MMCV's CUDA deformable-attention op has no BF16 kernel in this build.
    # A sub-unit FP16 loss scale reduces backward intermediate range while
    # unscaling to the same final gradient before clipping and optimization.
    dtype="float16",
    loss_scale=0.25,
    accumulative_counts=1,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        type="AdamWScheduleFreeOptimizer",
        lr=base_lr,
        weight_decay=1e-4,
        warmup_steps=0,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # Both backbones start from useful weights (official B1 and the
            # transplanted B0), so update them below the topology-recovery LR.
            "backbone.rgb_backbone": dict(lr_mult=0.1),
            "backbone.depth_backbone": dict(lr_mult=0.1),
            # These paths are deliberately reinitialized by the converter.
            "backbone.depth_adapters": dict(lr_mult=1.0),
            "backbone.depth_beta": dict(lr_mult=1.0),
            "neck": dict(lr_mult=1.0),
            # FFNs are physically sliced but otherwise teacher-initialized.
            "encoder": dict(lr_mult=0.5),
            "decoder": dict(lr_mult=0.5),
            # Detection ranks must adapt to the joint 800x600 distribution.
            "bbox_head.cls_branches": dict(lr_mult=1.0),
            "bbox_head.reg_branches": dict(lr_mult=1.0),
            # Reused metric-pose predictors receive a smaller recovery step.
            "bbox_head.reg_centers_2d_branch": dict(lr_mult=0.25),
            "bbox_head.reg_z_branch": dict(lr_mult=0.25),
            "bbox_head.reg_size_branch": dict(lr_mult=0.25),
            "bbox_head.reg_rotation_branch": dict(lr_mult=0.25),
            "bbox_head.cop_": dict(lr_mult=0.25),
            "bbox_head.depth_query_sampler": dict(lr_mult=0.25),
        },
    ),
)

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
param_scheduler = []
auto_scale_lr = dict(enable=False, base_batch_size=30)
randomness = dict(seed=20260826, deterministic=False)

load_from = None
resume = False
