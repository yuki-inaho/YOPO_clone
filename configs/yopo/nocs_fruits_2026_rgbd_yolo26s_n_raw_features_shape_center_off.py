"""F-off: exact F-on control with only the KLD centre term removed."""

_base_ = [
    "./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_shape_center_on.py"
]

model = dict(bbox_head=dict(loss_ellipsoid=dict(include_center=False)))
work_dir = "work_dirs/yopo_shape_center_off_20260906"
