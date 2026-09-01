"""Spend more of the objective on ellipse geometry, which is where 2D is lost.

The threshold sweep says where the 2D metric goes:

    recall @ IoU 0.25 = 0.8554     objects actually found
    recall @ IoU 0.50 = 0.7865
    recall @ IoU 0.75 = 0.2406     ellipses that are genuinely tight
    mean matched rIoU = 0.70

So 14.5% of annotations are missed outright, but a further 6.9% *are* found and
lose the match only because the predicted ellipse is too loose -- and among the
ones that do match, the envelope overlap averages 0.70 where a perfect
prediction would give 1.0.  ``mAP@0.25`` is already 0.7930 against a 0.8170
target, which means most of the remaining gap is ellipse accuracy rather than
detection.  (The metric itself was checked against an independent polygon
implementation over 12k pairs and is exact, so the 0.70 is real error.)

The ellipse KLD currently carries weight 2.0 while the horizontal box carries
5.0 (L1) plus 2.0 (GIoU).  The branch that the reported metric reads is the
cheapest term in the objective.  This run raises it to 6.0 and changes nothing
else.
"""

_base_ = ["./gaucho_800x600_base.py"]

model = dict(
    bbox_head=dict(
        loss_ellipsoid=dict(include_center=False),
        loss_ellipse2d=dict(loss_weight=6.0),
    ),
)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
