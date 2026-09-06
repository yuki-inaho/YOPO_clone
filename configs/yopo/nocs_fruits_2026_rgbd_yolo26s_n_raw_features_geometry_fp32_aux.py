"""E: isolate AMUSE auxiliary geometry outputs on top of FP32 geometry."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_fp32.py"]

model = dict(bbox_head=dict(geometry_output_aux_optimizer=True))

work_dir = "work_dirs/yopo_geometry_fp32_aux_20260906"
