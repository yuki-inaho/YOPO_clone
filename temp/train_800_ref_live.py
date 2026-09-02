"""Let the ellipse objective reach the box branch it is built on.

The 2D ellipse decodes as ``box_centre + offset * extent``, and the reference
box has always been detached.  The stated reason -- that matching must not
reshape the box branch through the ellipse chart -- is right for the assigner
and was carried into the regression loss as well, where it has a cost: the
ellipse objective cannot correct the centre it inherits, so the offset has to
absorb the box centre's error without being able to observe it.

It measurably does not absorb it.  Three independent estimates of the same
point agree to a thousandth::

    predicted box centre   0.0867
    predicted centers_2d   0.0902
    predicted ellipse      0.0866      (n = 5,999, in units of object size)

The ellipse centre is the box centre.  That is also where the AP is: a partial
oracle puts the centre at +0.0637 mAP50 against +0.0191 for extent and +0.0019
for orientation, and thirteen changes confined to the ellipse chart -- higher
weight, matching cost, corner and geodesic-angle terms, an explicit centre
loss, more queries, more resolution, a dense auxiliary head -- moved it by at
most +0.0053.  They could not have done better: none of them could reach the
tensor that decides the answer.

This removes the detach on the regression path only.  The assigner keeps its
own detached decode, since a gradient through matching has no business
existing.  The risk is the mirror of the benefit -- the ellipse objective can
now distort the box branch that feeds detection, so ``bbox``/``iou`` losses and
recall are the things to watch, not just the ellipse number.

Built on the current 2D best so the depth anchor, its window and the centre
term are unchanged and the difference is attributable to the gradient path.
"""

_base_ = ["./train_800_centre.py"]

model = dict(
    bbox_head=dict(
        gaucho_ellipse2d_reference_detach=False,
    ),
)
