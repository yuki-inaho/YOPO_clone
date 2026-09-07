"""Distance-prioritized soft-affinity online tracking ablation."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_sequence_residual_curriculum_full.py"]

sequence_context = dict(
    inference=dict(
        tracker=dict(
            assignment_strategy="soft_affinity_v1",
            geometry_gate_m=0.15,
            geometry_affinity_sigma_m=0.05,
            embedding_affinity_temperature=0.20,
            geometry_weight=0.70,
            embedding_weight=0.30,
        ),
    ),
)
