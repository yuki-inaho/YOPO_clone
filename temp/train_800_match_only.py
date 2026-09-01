"""Separate the two halves of the block that won 3D but cost 2D.

Run A changed the Hungarian cost *and* the matchability quality target in one
step, on the advice that they belong together.  It gained 16% on shared AP20
and lost 0.016 of rotated-NMS mAP50, and as written that trade cannot be
attributed to either half.

This run keeps the ellipse in the matching and puts the quality target back to
``hbb_iou``.  Against run A it isolates the ranking change; against the
baseline it isolates the matching change.
"""

_base_ = ["./train_800_best_latest.py"]

model = dict(
    bbox_head=dict(
        quality_target=dict(
            _delete_=True,
            type="MatchabilityQualityPolicy",
            source="hbb_iou",
        ),
    ),
)
