"""Stage 5: make CoP the inference path and introduce GWD consistency."""

_base_ = ['./nocs_fruits_2026_rgbd_shared_stage4_cop_aux.py']

model = dict(
    bbox_head=dict(
        cop_prediction_mode='chain',
        cop_encoder_pose_supervision=False,
        cop_obb_rotation_refinement=True,
        loss_z=dict(loss_weight=50.0),
        loss_sizes=dict(loss_weight=50.0),
        loss_rotation=dict(loss_weight=5.0),
        loss_obb_aux=dict(
            _delete_=True,
            type='GaussianGWDLoss',
            loss_weight=1.0,
            tau=1.0,
            normalize=True,
            include_center=False,
            fail_on_invalid=True,
        ),
        loss_projection=dict(
            _delete_=True,
            type='ProjectedEllipsoidGWDLoss',
            loss_weight=0.25,
            tau=1.0,
            normalize=True,
            detach_center=True,
            detach_depth=True,
            detach_size=True,
            fail_on_invalid=True,
        ),
    ),
)

optim_wrapper = dict(
    optimizer=dict(lr=5e-5, aux_lr=5e-5),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.cop_': dict(lr_mult=0.25),
            'bbox_head.depth_query_sampler': dict(lr_mult=0.25),
        },
    ),
)

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
load_from = None
resume = False
