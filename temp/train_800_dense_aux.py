"""A dense auxiliary head on the shared features -- the last identified lever.

Everything cheaper has been measured and is exhausted.  The 2D headroom is the
ellipse centre (+0.064 mAP50, against +0.019 extent and +0.001 angle), and the
centre does not move: three *independent* estimates of it -- the predicted box
centre, ``centers_2d``, and the ellipse itself -- all sit at 0.087 of object
scale.  Three heads converging on the same error is not a head problem; it is
the positional precision of the representation they share.

A one-to-one matcher gives each object exactly one positive, so that
representation is trained on ~20 supervised locations per image.  The reference
detector this work is measured against is dense: its DynamicSoftLabel
assignment gives each object thirteen, and it reaches 0.851 where this reaches
0.7365.  So the same supervision is attached here to the neck features, with
the assigner and the k the reference uses.

Training only.  Inference still reads the one-to-one head; nothing about the
deployed path changes.  The auxiliary is supervised on the *oriented*
annotation, decoded from the same compact Gaussian the ellipse branch uses, so
it is helping with the target that is actually scored rather than a horizontal
approximation of it.

Built on the current best model, so the auxiliary is the only difference.
"""

_base_ = ["./train_800_centre.py"]

model = dict(
    dense_aux_loss_weight=1.0,
    dense_aux_head=dict(
        type="RotatedRTMDetSepBNHead",
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 16, 32, 64],
        norm_cfg=dict(type="GN", num_groups=32),
        # The classification term came out at 20.5 against a 292 total in the
        # smoke step -- it normalises over dense locations, not over queries --
        # so it is scaled down to sit alongside the main objective rather than
        # dominate it.  The auxiliary is here to shape the features, not to
        # take over the gradient.
        loss_cls=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0,
                      loss_weight=0.1),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_iou=dict(type="RotatedIoULoss", mode="linear", loss_weight=2.0),
    ),
)

max_epochs = 12
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=1)
