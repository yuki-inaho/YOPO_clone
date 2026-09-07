"""Residual context curriculum over prediction-derived G10 observations."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_sequence_prediction_full.py"]

sequence_context = dict(
    work_dir="work_dirs/yopo_sequence_residual_curriculum_20260907",
    head=dict(fusion_strategy="residual_gated_v1"),
    training=dict(strategy="residual_context_curriculum_v1"),
    evaluation=dict(
        comparison_contract=(
            "shared_B0_B2_then_zero_gate_parent_C0_C1_prediction_observations_v1"
        ),
    ),
)
