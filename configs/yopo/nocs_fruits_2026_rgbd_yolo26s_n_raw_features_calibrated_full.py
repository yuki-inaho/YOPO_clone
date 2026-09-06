"""Raw s/n backbone+encoder stacks with train-only calibrated YOPO interfaces."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_full.py"]

load_from = "work_dirs/yolo26s_n_raw_features_calibrated.pth"
train_dataloader = dict(batch_size=25)
work_dir = "work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906"
