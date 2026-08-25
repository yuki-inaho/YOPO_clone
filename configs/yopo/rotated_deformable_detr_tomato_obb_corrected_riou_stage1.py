"""Phase 1: RGB-only rotated-IoU adaptation on corrected tomato OBBs."""

_base_ = ["../_base_/default_runtime.py"]

custom_imports = dict(
    imports=[
        "yopo.datasets.dota_tomato",
        "yopo.engine.optimizers.deim_optimizers",
        "yopo.evaluation.metrics.rotated_iou_metric",
    ],
    allow_failed_imports=False,
)

data_root = (
    "/workspace/YOPO_clone/datasets/"
    "fruit_obb_o2deim-stage3sf-cvat-corrected_"
    "train1368-box94605_test181-box9034_20260704/"
)
dataset_metainfo = dict(classes=("tomato",), palette=[(255, 255, 0)])
backend_args = None

model = dict(
    type="DeformableDETR",
    num_queries=150,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False,
        non_blocking=True,
    ),
    backbone=dict(
        type="HGNetV2",
        name="B2",
        in_channels=3,
        return_idx=[1, 2, 3],
        # Freeze only the stem. HGNet stages, neck, transformer and OBB head
        # remain trainable for RGB-domain adaptation.
        freeze_at=0,
        freeze_stem_only=True,
        freeze_norm=False,
        init_cfg=None,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=4,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1,
            ),
        ),
    ),
    decoder=dict(
        num_layers=4,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            ),
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1,
            ),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="RotatedDeformableDETRHead",
        num_classes=1,
        angle_cfg=dict(width_longer=True, start_angle=0),
        angle_factor=3.141592653589793,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(
            type="RotatedIoULoss",
            mode="linear",
            loss_weight=2.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(
                    type="RBoxL1Cost",
                    weight=5.0,
                    box_format="xywha",
                    angle_factor=3.141592653589793,
                ),
                dict(
                    type="GDCost",
                    loss_type="kld",
                    fun="log1p",
                    tau=1,
                    sqrt=False,
                    weight=2.0,
                ),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=150),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, box_type="rbox"),
    # Native-size identity resize supplies scale_factor without changing OBB
    # geometry and keeps train/validation transforms symmetric.
    dict(type="Resize", scale=(800, 600), keep_ratio=False),
    dict(
        type="RandomFlip",
        prob=0.5,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="PackDetInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, box_type="rbox"),
    dict(type="Resize", scale=(800, 600), keep_ratio=False),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="DOTAOBBDataset",
        data_root=data_root,
        ann_file="train/labels",
        data_prefix=dict(img_path="train/images"),
        metainfo=dataset_metainfo,
        img_shape=(600, 800),
        img_suffixes=(".jpg",),
        strict_loading=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DOTAOBBDataset",
        data_root=data_root,
        ann_file="test/labels",
        data_prefix=dict(img_path="test/images"),
        metainfo=dataset_metainfo,
        img_shape=(600, 800),
        img_suffixes=(".jpg",),
        strict_loading=True,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    type="yopo.engine.optimizers.deim_optimizers.AmpScheduleFreeOptimWrapper",
    constructor="DefaultOptimWrapperConstructor",
    dtype="float16",
    loss_scale=1.0,
    optimizer=dict(
        type=(
            "yopo.engine.optimizers.deim_optimizers."
            "MuonScheduleFreeOptimizer"
        ),
        muon_lr=1e-4,
        muon_weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        sf_lr=5e-6,
        sf_betas=[0.9, 0.95],
        sf_eps=1e-8,
        sf_weight_decay=0.000125,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

max_epochs = 5
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
val_evaluator = dict(
    type="RotatedIoUMetric",
    iou_thr=0.5,
    score_thr=0.05,
    num_classes=1,
)
test_evaluator = val_evaluator
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4],
        gamma=0.1,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        save_last=True,
        max_keep_ckpts=2,
        save_best="rbbox_mAP_50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=10),
)

load_from = (
    "work_dirs/checkpoint_expansions/"
    "rddetr_tomato_riou_epoch20_q150.pth"
)
resume = False
