"""QA-O2O: localization-quality score with one-to-one NMS-free queries."""

_base_ = ["./nocs_custom_fruit_rgbd_nmsfree_control_continue5.py"]

# The sole ablation change is binary FocalLoss -> aligned-IoU QualityFocalLoss.
# Hungarian assignment, query count, geometry heads, optimizer and schedule are
# identical to the time control.  Positive IoU targets are detached in the head.
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    ),
)
