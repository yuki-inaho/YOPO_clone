"""Jointly fine-tune parallel and CoP pose paths on the TMT4-02 export.

The 450/50 split is already materialized as native 800x600 RGB, mapped depth
and portable 3D OBB labels.  This config deliberately keeps the proven
Stage-4 2D/3D consistency objectives and performs model-only transfer:
optimizer state from the 2025+2026 curriculum is not valid for this smaller
target dataset.

Set ``YOPO_TMT4_02_DATA_ROOT`` or ``YOPO_TMT4_02_SOURCE_CHECKPOINT`` to move
the data/artifacts without editing this config.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_compact_curriculum_"
    "stage4_cop_chain_gwd_full.py"
]

data_root = "{{$YOPO_TMT4_02_DATA_ROOT:/home/kasm-user/Downloads/2026-08-04_tmt4-02_subsample_500_yopo_800x600_train450-valid50/}}"
source_checkpoint = "{{$YOPO_TMT4_02_SOURCE_CHECKPOINT:/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/compact_joint800_curriculum_stage4_cop_chain_gwd_b30_from_cume5_bf16_fp32attn_e15/best_3d_iou_0.25_epoch_5.pth}}"

# ``auxiliary`` keeps the parallel pose prediction as the primary validation
# stream and trains CoP on the identical 2D-Hungarian query/GT assignments.
# Splitting the former Stage-4 pose budget evenly avoids doubling its scale.
# The inherited projected-ellipsoid GWD constrains parallel 3D rotation against
# the target 2D OBB; the inherited query-Gaussian GWD constrains the CoP OBB
# representation and refines the CoP rotation stage.
model = dict(
    bbox_head=dict(
        cop_prediction_mode="auxiliary",
        cop_encoder_pose_supervision=True,
        cop_aux_loss_weights=dict(z=1.0, size=1.0, rotation=1.0),
        cop_obb_rotation_refinement=True,
        loss_z=dict(loss_weight=25.0),
        loss_sizes=dict(loss_weight=25.0),
        loss_rotation=dict(loss_weight=2.5),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_z_branch": dict(lr_mult=0.5),
            "bbox_head.reg_size_branch": dict(lr_mult=0.5),
            "bbox_head.reg_rotation_branch": dict(lr_mult=0.5),
            "bbox_head.cop_": dict(lr_mult=0.5),
            "bbox_head.depth_query_sampler": dict(lr_mult=0.5),
        },
    ),
)

# Keep the native identity-geometry pipelines inherited from the 2025+2026
# config.  ``custom_val`` is intentional: it reads ``real/test_list.txt`` but
# consumes the train-style per-frame ``*_label.pkl`` files in this export.
train_dataloader = dict(
    _delete_=True,
    batch_size=30,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="NOCSCustomFruitDataset",
        data_root=data_root,
        split="real_train",
        obb_coordinate_scale=1.0,
        pipeline={{_base_.native_train_pipeline}},
        backend_args=None,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="NOCSCustomFruitDataset",
        data_root=data_root,
        split="custom_val",
        obb_coordinate_scale=1.0,
        pipeline={{_base_.native_val_pipeline}},
        backend_args=None,
    ),
)

# 450 images / batch 30 = 15 updates per epoch.  Thirty epochs therefore give
# 450 target-domain updates while validation every five epochs catches both
# rapid adaptation and regression.  The Stage-4 stable BF16 LR remains 5e-5.
max_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        save_last=True,
        max_keep_ckpts=3,
        save_best=["3d_iou_0.25", "AP50_95", "AP75"],
        rule=["greater", "greater", "greater"],
        save_optimizer=True,
    ),
)

custom_hooks = [
    dict(type="ScheduleFreeOptimizerModeHook"),
    dict(
        type="EarlyStoppingHook",
        monitor="3d_iou_0.25",
        rule="greater",
        min_delta=5e-4,
        patience=4,
        strict=True,
        check_finite=True,
    ),
]

load_from = source_checkpoint
resume = False
