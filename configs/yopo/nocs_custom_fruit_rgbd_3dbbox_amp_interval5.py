"""The measured long-run speed policy: evaluate full NOCS validation every 5 epochs."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"]

# Keep the model, batch 26, fp16 AMP, optimizer and NOCSMetric unchanged.
# The measured 16 s evaluator/metric aggregation is run once per five train
# epochs; the actual metric remains a full 50-image validation at each gate.
train_cfg = dict(val_interval=5)
