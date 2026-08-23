_base_ = ['./nocs_custom_real_hgnetv2_rgbd.py']

# DEIM-style experiment-management config for the custom RGB-D fruit dataset.
#
# Inherits the HGNetV2 (B2) + RGB-D 4-channel model + data setup from
# ``nocs_custom_real_hgnetv2_rgbd.py`` (and its base chain, which ultimately
# pulls ``../_base_/default_runtime.py``) and overrides the training-stack
# pieces to match the DEIM workflow:
#   - ScheduleFree optimizer (AdamWScheduleFreeOptimizer) wrapped by
#     ``ScheduleFreeOptimWrapper`` (keeps the optimizer in train mode, which
#     schedulefree requires).
#   - encoder lr*0.5 via paramwise_cfg (AutoMuon was dropped because its ndim
#     re-grouping silently ignores paramwise cfg; see the workdoc §4 for the
#     verification log).
#   - Effective batch 24 via batch_size=8 + accumulative_counts=3. Note: fp16
#     AMP is intentionally NOT used — a fresh fp16 forward produced assigner
#     NaNs with this custom model, and fp32 batch-8 fits in ~8.7 GB < 20 GB.
#   - TensorBoard (vis_backends includes TensorboardVisBackend, on top of the
#     default LocalVisBackend)
#   - TopKCheckpointHook (keep best-K checkpoints ranked by 3d_iou_0.50)

custom_intrinsic = [443.9066, 449.1953, 321.3503, 230.8687]

# ── Hooks: TopK checkpoint (K-best pickup) ─────────────────────────────────
# Replace the default periodic CheckpointHook with TopKCheckpointHook, which
# keeps the best-3 checkpoints by 3d_iou_0.50 in addition to the managed
# best/last checkpoints.
default_hooks = dict(
    checkpoint=dict(
        type='TopKCheckpointHook',
        interval=1,
        topk=3,
        key_indicator='3d_iou_0.50',
        rule='greater',
        save_best='3d_iou_0.50'))

# ── TensorBoard ────────────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ── Optimizer: ScheduleFree optimizer (AdamWScheduleFree) ────────────────────
# NOTE: AutoMuonWithAuxAdam re-partitions params by ndim (2D/4D -> Muon, rest
# -> AdamW), so mmengine's name-based ``paramwise_cfg`` (encoder lr_mult=0.5)
# is silently ignored with it. ScheduleFree is a plain torch optimizer whose
# per-group ``lr`` is honoured by the DefaultOptimWrapperConstructor, so we use
# it for the encoder lr*0.5 requirement (verified by printing param_groups).
# Backbone lr_mult=0.1 additionally slows down the pretrained HGNetV2 stem
# (matching the base ``nocs_custom_real_hgnetv2_rgbd`` fine-tune schedule).
optim_wrapper = dict(
    type='ScheduleFreeOptimWrapper',
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1),
                         encoder=dict(lr_mult=0.5))),
    optimizer=dict(
        type='AdamWScheduleFreeOptimizer',
        lr=0.0025,
        warmup_steps=0),
    accumulative_counts=3,
    clip_grad=dict(max_norm=0.1, norm_type=2))

# ── Learning policy (train until loss converges, cosine annealing) ──────────
# 20 epochs was not enough (loss 682->151 still slowly decreasing). Extended to
# 100 epochs (~75 min at ~45 s/epoch) with cosine-annealing lr so the loss can
# fully converge; resume from the 20-epoch checkpoint.
#   - Linear warmup (5 epochs, by iter) then cosine decay over the full run.
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# ── Dataloader: effective batch 24 (8 x 3 gradient accumulation) ───────────
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True),
]

# ── AMP (fp16) is intentionally NOT used (see header note); ScheduleFree runs
# at fp32 inside ScheduleFreeOptimWrapper. Do not pass --amp on the CLI.
