"""Teacher-free inference for the Q150 2D-first Stage 4 checkpoint."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage4_full.py"]

model = dict(
    bbox_head=dict(
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
resume = False
