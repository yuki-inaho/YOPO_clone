"""Teacher-free diagnostic inference for the 2D OBB foundation."""

_base_ = ["./nocs_custom_fruit_rgbd_2d_obb_foundation_full.py"]

model = dict(
    bbox_head=dict(
        expose_obb_aux_predictions=True,
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
