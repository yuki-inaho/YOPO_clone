# Smoke-test config: 1-epoch training on synthetic NOCS data.
#
# Inherits from the full training config and overrides the parts needed
# to make a single forward+backward pass succeed on a single GPU with
# no downloaded checkpoint and no dataloader workers.
#
# Usage (from repo root):
#   just smoke-train                              # preferred: work_dir -> work_dirs/smoke_train
#   python tools/train.py temp/smoke_nocs_r50_1iter.py   # direct: work_dir -> work_dirs/smoke_nocs_r50_1iter
#
# NOTE: `just smoke-train` passes --work-dir work_dirs/smoke_train, which
#   overrides the work_dir set at the bottom of this file. Running tools/train.py
#   directly (no --work-dir) uses this file's work_dir (work_dirs/smoke_nocs_r50_1iter).
#
# Expected outcome:
#   - One epoch completes against the 4-frame synthetic dataset.
#   - A checkpoint (epoch_1.pth) is saved under the active work_dir.
#
# Generate the synthetic data first:
#   python scripts/gen_synthetic_nocs.py --out data/nocs_smoke --n 4

_base_ = ['../configs/yopo/nocs_yopo_real_camera_r50_rgbd.py']

# ── Data root ─────────────────────────────────────────────────────────────────
# Point to synthetic data produced by scripts/gen_synthetic_nocs.py.
data_root = 'data/nocs_smoke/'

# ── Pretrained weights ────────────────────────────────────────────────────────
# Prevent downloading the large DINO COCO checkpoint.
load_from = None

# NOTE on backbone.init_cfg:
#   The base config has backbone.init_cfg = dict(type='Pretrained',
#   checkpoint='torchvision://resnet50').  We intentionally keep this so
#   the backbone weights are fetched from torchvision (small download ~100MB,
#   cached after first run).  This exercises the real weight-loading path.
#
#   To run fully offline (random weights, slower convergence but zero I/O):
#     model = dict(backbone=dict(init_cfg=None))

# ── Training loop ─────────────────────────────────────────────────────────────
# Base config uses EpochBasedTrainLoop; keep it.  One epoch is enough.
max_epochs = 1
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
)

# ── Disable validation (all three must be None together) ─────────────────────
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

# ── Pipelines ─────────────────────────────────────────────────────────────────
# These are inherited verbatim from the base config and re-declared here only
# so that the train_dataloader and val_dataloader overrides below can reference
# them by name within this file.  mmengine resolves these after inheritance.
# (No actual change to pipeline contents.)
backend_args = None
scale = (640, 480)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args,
         to_float32=True),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomTranslatePixels', prob=0.5, max_translate_offset=50),
    dict(type='RandomFlipFor9DPose', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pack9DPoseInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args,
         to_float32=True),
    dict(type='Load9DPoseAnnotations', with_bbox=True,
         with_centers_2d=True, with_z=True),
    dict(type='ConcatDepthToImage'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'intrinsic', 'models_info_path'),
    ),
]

# ── DataLoaders ───────────────────────────────────────────────────────────────
# batch_size=2 exercises collation (>1 sample) without consuming much memory.
# num_workers=0 avoids forking issues in minimal environments.
# persistent_workers must be False when num_workers=0.
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='NOCSDataset',
                data_root=data_root,
                split='camera_train',
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type='NOCSDataset',
                data_root=data_root,
                split='real_train',
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
        ],
    ),
)

# ── Hooks ─────────────────────────────────────────────────────────────────────
# Override logger to print every iteration (useful for smoke diagnostics).
# Checkpoint at interval=1 ensures a .pth is written after the single epoch.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# ── LR scheduler ─────────────────────────────────────────────────────────────
# Keep MultiStepLR (same type as the base config) but move the milestone
# beyond max_epochs=1 so the LR never actually decays during the smoke run.
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9999],   # unreachable with max_epochs=1
        gamma=0.1,
    )
]

# ── Work directory ────────────────────────────────────────────────────────────
work_dir = 'work_dirs/smoke_nocs_r50_rgbd_1iter'
