"""Give each object more than one positive, at the k the reference uses.

The 2D headroom is localisation: replacing the matched predictions' centres
with the annotations' is worth +0.064 mAP50, against +0.019 for extent and
+0.002 for angle.  The centre error sits at 1.5 px on 17 px objects and has not
moved under any loss change tried, which points at how much supervision each
object gets rather than at what that supervision says.

That is the structural difference from the reference: it is a dense detector
whose DynamicSoftLabel assignment gives each object thirteen positives, while a
one-to-one matcher gives it exactly one.  This head already carries a
one-to-many auxiliary (`o2m_aux_topk`, off by default); the previous attempt at
it used top-2 and lost 0.001, which is a different experiment from top-13.

Built on the current best model, so only the auxiliary is new.  Its weight is
left at the default 0.5: this is an auxiliary on the shared features, and
inference still reads the one-to-one head.
"""

_base_ = ["./train_800_centre.py"]

model = dict(
    bbox_head=dict(
        o2m_aux_topk=13,
        o2m_aux_loss_weight=0.5,
    ),
)
