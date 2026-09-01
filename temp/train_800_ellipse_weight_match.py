"""Ellipse weight 6.0 *and* the ellipse-aware matching cost.

The two levers that won their own comparisons, combined: the heavier ellipse
term for 2D geometry, and the ellipse in the Hungarian cost, which gained 16%
on shared AP20.  Run against the weight-only variant it also measures whether
the matching cost still costs 2D once the ellipse itself is better fitted --
the earlier 2D loss may have been a symptom of matching on a branch that was
not yet accurate.
"""

_base_ = ["./train_800_ellipse_weight.py"]

_hbb_costs = [
    dict(type="FocalLossCost", weight=2.0),
    dict(type="BBoxL1Cost", box_format="xywh", weight=5.0),
    dict(type="IoUCost", iou_mode="giou", weight=2.0),
]

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type="HungarianAssigner",
            match_costs=_hbb_costs + [dict(type="Ellipse2DKLDCost", weight=1.0)],
        ),
        encoder_assigner=dict(
            _delete_=True,
            type="HungarianAssigner",
            match_costs=_hbb_costs,
        ),
    ),
)
