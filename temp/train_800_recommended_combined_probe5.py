"""Combined probe to run only after both isolated probes pass their gates."""

_base_ = ["./train_800_ellipse_aware_probe5.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid_max_axis=dict(
            type="EllipsoidMaxAxisLoss",
            max_diameter=0.05,
            loss_weight=0.25,
        ),
        ellipsoid_gt_max_diameter=0.05,
        ellipsoid_max_diameter=0.05,
    ),
)

default_hooks = dict(
    checkpoint=dict(save_best="gaucho3d/shared_AP_20", rule="greater"),
)
