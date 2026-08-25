"""Teacher-free inference for anisotropy-weighted projection GWD."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_anisotropy.py"
]

model = dict(
    bbox_head=dict(
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
resume = False
