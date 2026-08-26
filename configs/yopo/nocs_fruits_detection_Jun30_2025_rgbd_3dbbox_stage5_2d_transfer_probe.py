"""Gate a coherent pretrained DINO 2D path on the portable 3D model.

The source reached AP50 0.334 on its original NOCS validation split.  Only its
encoder, decoder, queries, two-stage proposal transform, class branches, and
2D box branches are overlaid.  The portable RGB-D backbone/neck and every 3D
pose branch remain from the Stage-4 IoU50 checkpoint.
"""

_base_ = [
    './nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_'
    'stage5_detection_repair_probe.py'
]

detection_checkpoint = (
    'work_dirs/nocs_custom_fruit_rgbd_2d_foundation_full/'
    'best_AP50_epoch_50.pth'
)

custom_hooks = [
    dict(
        type='DetectionRepairTransferHook',
        checkpoint=detection_checkpoint,
        report_filename='detection_repair_transfer_report.json',
        query_seed=736512,
    ),
]
