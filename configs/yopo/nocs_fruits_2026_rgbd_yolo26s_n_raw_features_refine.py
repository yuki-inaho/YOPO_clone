"""Low-LR joint refinement from the validated raw-feature stage-B best."""

_base_ = ["./nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py"]

optim_wrapper = dict(optimizer=dict(lr=1e-5, aux_lr=1e-5))
max_epochs = 15
train_cfg = dict(max_epochs=max_epochs)
# Keep all three validation candidates until the joint 2D/3D guard is checked.
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
load_from = (
    "work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/"
    "best_ellipsoid_shared_AP_25_epoch_20.pth"
)
resume = False
work_dir = "work_dirs/yopo_yolo26s_n_raw_features_refine_20260906"
