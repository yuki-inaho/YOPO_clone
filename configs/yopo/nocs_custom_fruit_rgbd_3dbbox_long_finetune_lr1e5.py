"""Single-factor long fine-tune candidate after the LR=5e-5 gate regressed."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_long_finetune.py"]

# The first fresh-optimizer gate lost 43.7% IoU@0.50 after five epochs at
# 5e-5.  Change only the optimizer LR; keep source weights, schedule horizon,
# batch/AMP, frozen RGB, depth MAE, evaluator and checkpoint policy identical.
optim_wrapper = dict(optimizer=dict(lr=1e-5))
