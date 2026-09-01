"""Continue from the best checkpoint with ellipse-aware matching and ranking.

Everything before this point selected and ranked queries on the horizontal box
alone -- focal score, L1, GIoU -- and then applied the ellipse loss to whatever
that picked.  The reported metric is a rotated ellipse envelope, so the two
decisions that matter most, *which* query owns an annotation and *how confident*
it is, were being made by a criterion blind to orientation and elongation.

Two changes, one block:

* ``Ellipse2DKLDCost`` joins the decoder's Hungarian costs, so matching sees the
  ellipse.  Symmetric KLD, so a query cannot win by predicting something large
  and vague that merely covers the annotation.
* The matchability quality target moves from ``hbb_iou`` to
  ``ellipse_kld_blend``, so the score the detector learns to emit reflects
  ellipse agreement rather than box overlap alone.

The encoder assigner keeps the horizontal costs: encoder proposals have no
ellipse branch to compare, which is also why the quality policy is given
``missing_obb='hbb_iou'`` as an explicit fallback rather than being allowed to
fail on that path.

The 3D side is unchanged from the run this continues -- ``include_center=False``
and nothing else -- so any movement is attributable to the two changes above.
The physical 5 cm bound stays off here; it is a separate ablation on purpose.
"""

_base_ = ["./gaucho_800x600_base.py"]

_hbb_costs = [
    dict(type="FocalLossCost", weight=2.0),
    dict(type="BBoxL1Cost", box_format="xywh", weight=5.0),
    dict(type="IoUCost", iou_mode="giou", weight=2.0),
]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
        quality_target=dict(
            _delete_=True,
            type="MatchabilityQualityPolicy",
            source="ellipse_kld_blend",
            obb_weight=0.5,
            include_center=True,
            # Encoder proposals reach the policy without an ellipse; say so
            # rather than letting it raise mid-epoch.
            missing_obb="hbb_iou",
        ),
    ),
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

# Warm start from a trained ellipsoid branch, so a shorter run is informative.
max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
