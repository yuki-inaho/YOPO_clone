"""Teacher-free inference for the OBB-foundation Stage-4 control."""

_base_ = ["./nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_control.py"]

model = dict(
    bbox_head=dict(
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
