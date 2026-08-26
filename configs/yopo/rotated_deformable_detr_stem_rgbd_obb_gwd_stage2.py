"""GWD refinement of the RGB-D stem OBB detector."""

_base_ = ["./rotated_deformable_detr_stem_rgbd_obb_riou_stage1.py"]

model = dict(
    bbox_head=dict(
        loss_iou=dict(
            _delete_=True,
            type="GDLoss",
            loss_type="gwd",
            fun="log1p",
            tau=1,
            loss_weight=2.0,
        ),
    ),
)

optim_wrapper = dict(
    optimizer=dict(
        muon_lr=5e-5,
        sf_lr=2.5e-6,
    ),
)

max_epochs = 15
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[12],
        gamma=0.1,
    ),
]

# Stage 2 loads the complete stage-1 RGB-D state and must not replay RGB-only
# transfer over the learned fusion/depth adaptation.
custom_hooks = []
load_from = (
    "work_dirs/rddetr_stem_rgbd_obb_riou_stage1/selected_best.pth"
)
resume = False
