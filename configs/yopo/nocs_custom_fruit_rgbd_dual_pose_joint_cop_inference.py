"""Teacher-free CoP-path inference for the same joint checkpoint."""

_base_ = ["./nocs_custom_fruit_rgbd_dual_pose_joint.py"]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_encoder_pose_supervision=False,
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)
load_from = None
resume = False
