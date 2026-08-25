"""Teacher-free QA-O2O validation with opt-in OBB diagnostics."""

_base_ = ["./nocs_custom_fruit_rgbd_nmsfree_quality_focal.py"]

model = dict(
    bbox_head=dict(
        expose_obb_aux_predictions=True,
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
