"""Teacher-free time-control validation with opt-in OBB diagnostics."""

_base_ = ["./nocs_custom_fruit_rgbd_nmsfree_control_continue5.py"]

model = dict(
    bbox_head=dict(
        expose_obb_aux_predictions=True,
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
