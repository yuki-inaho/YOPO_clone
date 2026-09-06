"""YOLO26n RGB-D FULL training from the train-only calibrated boundary."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26n_stage1_full.py"]

# Weight-only initialization; optimizer and training state start fresh.
load_from = "work_dirs/yolo26n_rgbd_frontend_calibrated_train64.pth"
resume = False
