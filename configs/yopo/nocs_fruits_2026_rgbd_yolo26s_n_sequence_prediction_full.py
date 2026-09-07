"""Prediction-derived sequence descriptor curriculum from G10 raw weights."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_sequence_context.py"]

sequence_context = dict(
    work_dir="work_dirs/yopo_sequence_prediction_20260907",
    feature_extraction=dict(
        source="frozen_g10_detector_prediction_hbb_center",
        cache_dtype="float16",
        prediction_batch_probe_sizes=[1],
        parity_reference_batch_size=1,
        batch_policy="causal_batch1_parity_over_vram_utilization",
    ),
    prediction_matching=dict(
        score_threshold=0.30,
        max_detections=128,
        class_id_mapping={0: 1},
        min_iou=0.30,
        max_center_distance_px=16.0,
        assignment="hungarian_partial_no_fallback",
        teacher_geometry_role="supervision_only",
    ),
    evaluation=dict(
        comparison_contract="prediction_observations_same_seed_same_candidates_v1",
    ),
)
