"""Teacher-free inference config for a trained Stage 4 CoP checkpoint.

Pass the trained checkpoint with ``--checkpoint`` or ``--cfg-options
load_from=...``. Distillation affects training losses only, so removing the
teacher modules leaves the Stage 4 prediction architecture unchanged while
avoiding any dependency on the original teacher checkpoint files.
"""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_stage4_full.py"]

model = dict(
    bbox_head=dict(
        distill_attributes=(),
        obb_center_teacher_checkpoint=None,
        pose_teacher_checkpoint=None,
    ),
)

load_from = None
resume = False
