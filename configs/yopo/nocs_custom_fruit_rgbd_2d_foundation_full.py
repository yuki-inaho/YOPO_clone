"""Full 2D localization pretraining before any staged 3D pose training."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_curriculum_base.py"]

# The updated dataset contains as many as 147 objects in one training image
# and 115 in validation, so the historical 100-query cap is a real recall
# bottleneck. Keep a small explicit margin while retaining dense batching.
max_objects = 150
max_epochs = 50

load_from = "work_dirs/checkpoint_expansions/rgbd3d_epoch10_q150.pth"
resume = False

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        test_cfg=dict(max_per_img=max_objects),
        cop_prediction_mode="chain",
        cop_encoder_pose_supervision=False,
        cop_fusion_mode="query_dense",
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
        loss_z=dict(loss_weight=0.0),
        loss_sizes=dict(loss_weight=0.0),
        loss_rotation=dict(loss_weight=0.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        ),
    ),
)

# This is an encoder/decoder/head localization run, not the conservative 3D
# curriculum. RGB remains the verified frozen OBB source; the depth branch,
# transformer, and 2D prediction branches are trainable.
optim_wrapper = dict(optimizer=dict(lr=5e-5))
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=5,
        save_last=True,
        max_keep_ckpts=1,
        save_best="AP50",
        rule="greater",
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)
