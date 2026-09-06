"""G: isolate zero-init depth-query context for the GauCho shape head."""

_base_ = [
    "./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_fp32_aux.py"
]

model = dict(bbox_head=dict(gaucho_depth_context=True))

work_dir = "work_dirs/yopo_geometry_depth_shape_20260906"
