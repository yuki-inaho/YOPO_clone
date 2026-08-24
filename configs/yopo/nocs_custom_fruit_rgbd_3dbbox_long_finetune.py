"""Fresh long fine-tuning from the best finite RGB-D 3D checkpoint.

The runner applies ``load_from`` before ``before_train``.  The inherited
``RGBBackboneTransferHook`` then strict-loads only the frozen RGB branch from
the original 2D checkpoint.  That ordering is deliberate: the RGB tensors in
the full epoch-14 checkpoint are byte-identical to the frozen 2D source (and
the accompanying contract test proves it), while the depth/3D decoder/head
remain initialized from the selected full 3D checkpoint.
"""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_amp_interval5.py"]

# This is a fresh fine-tune, not a resume: checkpoint optimizer/scheduler
# state is intentionally not restored.  Keep the run separate from its source
# work directory so the source best checkpoint remains immutable evidence.
load_from = (
    "work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/"
    "best_3d_iou_0.50_epoch_14.pth"
)
resume = False

# Maintain the measured high-utilization batch/AMP path but reduce the base LR
# fourfold for continuation from a trained 3D solution.  Full validation is a
# 50-image quality gate every five epochs; only the best metric weights are
# retained by the inherited bounded checkpoint policy.
max_epochs = 80
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs,
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    ),
]

# Fixed seed makes the quality gate and long run directly comparable.  The
# deterministic flag remains false to preserve the validated high-throughput
# CUDA execution path.
randomness = dict(seed=3407, deterministic=False)
