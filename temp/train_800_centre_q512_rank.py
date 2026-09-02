"""Rank the queries by the ellipse the evaluation actually scores.

The 2D gap to the reference detector is coverage, not geometry: on the frames
the two validation sets share, the centre error is 2.09 px against 2.02 px and
the matched overlap 0.695 against 0.711, while recall is 0.753 against 0.830.
Raising the query count to 512 does supply the missing coverage -- the ceiling
with no threshold and no suppression goes 0.7822 -> 0.8212 -- but the score
cannot order what the extra queries produce, and mAP falls 0.7048 -> 0.6818
even though recall rises.

The reason is visible in the head's own configuration.  ``quality_target``
trains the classification score against ``hbb_iou``: the quality of the
*horizontal* box.  Evaluation ranks by the rotated IoU of the ellipse envelope.
The score has been ordering a different quantity from the one being measured,
which costs little at 256 queries -- where few duplicates compete -- and a
great deal at 512, where ordering is the whole problem.

``MatchabilityQualityPolicy`` already implements the ellipse sources and the
head already routes the decoded ellipse into them; nothing here is new
machinery, only a source that was never selected.  The blend keeps half the
weight on the HBB term, which is well conditioned from the first epoch, while
the ellipse term is still forming.

Encoder proposals carry no ellipse branch, so ``missing_obb='hbb_iou'`` makes
the fallback explicit rather than letting it raise.  ``include_center`` is on:
a duplicate displaced from its object should rank below the query that sits on
it, and this target is detached, so the centre term cannot drive the inflation
it causes inside a KLD *loss*.
"""

_base_ = ["./train_800_centre_q512.py"]

model = dict(
    bbox_head=dict(
        quality_target=dict(
            type="MatchabilityQualityPolicy",
            source="ellipse_kld_blend",
            obb_weight=0.5,
            missing_obb="hbb_iou",
            include_center=True,
            tau=1.0,
            fail_on_invalid=False,
        ),
    ),
)
