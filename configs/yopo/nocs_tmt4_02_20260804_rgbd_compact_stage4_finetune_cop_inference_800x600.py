"""Evaluate the CoP stream of the TMT4-02 dual-path fine-tune checkpoint."""

_base_ = [
    "./nocs_tmt4_02_20260804_rgbd_compact_stage4_"
    "finetune_800x600.py"
]

model = dict(
    bbox_head=dict(
        cop_prediction_mode="chain",
        cop_encoder_pose_supervision=False,
    ),
)

load_from = None
resume = False
