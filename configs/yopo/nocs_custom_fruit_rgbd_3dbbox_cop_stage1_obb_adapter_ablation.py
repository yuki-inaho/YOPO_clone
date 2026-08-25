"""Rejected-by-default OBB head-only center adapter ablation.

This config exists to keep the requested OBB self-distillation path
reproducible.  It is not part of the main curriculum: on three NOCS validation
images its frozen head-only center target was worse than the epoch10 student.
"""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_stage1_obb.py"]

model = dict(
    bbox_head=dict(
        center_teacher_source="obb_adapter",
        distill_score_threshold=0.0,
    ),
)
