"""FULL RIoU stage with a config-composable residual pyramid refiner."""

_base_ = ["./rotated_deformable_detr_stem_rgbd_obb_riou_stage1.py"]

custom_imports = dict(
    imports=[
        "yopo.datasets.dota_tomato",
        "yopo.datasets.transforms.raw_depth",
        "yopo.engine.hooks.rgbd_obb_transfer",
        "yopo.engine.optimizers.deim_optimizers",
        "yopo.evaluation.metrics.rotated_iou_metric",
        "yopo.models.backbones.dual_rgbd",
        "yopo.models.necks.composable_pyramid_neck",
    ],
    allow_failed_imports=False,
)

model = dict(
    neck=dict(
        _delete_=True,
        type="ComposablePyramidNeck",
        mapper=dict(
            type="ChannelMapper",
            in_channels=[256, 256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type="GN", num_groups=32),
            num_outs=4,
        ),
        refiner=dict(
            type="ResidualPyramidRefiner",
            channels=256,
            num_levels=4,
            beta_init=0.0,
        ),
    ),
)

custom_hooks = [
    dict(
        type="RGBDOBBTransferHook",
        checkpoint=(
            "work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/"
            "selected_best.pth"
        ),
        report_filename="rgbd_obb_transfer_report.json",
    ),
]
