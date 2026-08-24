"""One-epoch opt-in timing benchmark for the final RGB-D 3D BBOX contract."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"]

max_epochs = 1
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=3407, deterministic=False)

# This config deliberately preserves the production batch/AMP/data/evaluator
# contract.  It only disables checkpoint writes and adds synchronized timing.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=100,
        save_last=False,
        save_best=None,
        save_optimizer=False,
    ),
    logger=dict(interval=1),
)

custom_hooks = [
    dict(
        type="RGBBackboneTransferHook",
        checkpoint=(
            "work_dirs/rddetr_tomato_riou_linear_ft20/"
            "best_rbbox_mAP_50_epoch_20.pth"
        ),
        report_filename="partial_transfer_report.json",
    ),
    dict(
        type="ThroughputBenchmarkHook",
        output_filename="throughput_benchmark.json",
        warmup_iters=2,
    ),
]
