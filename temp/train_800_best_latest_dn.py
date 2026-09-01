"""Ellipse-aware matching and ranking, plus the ellipse loss on DN queries.

Identical to ``train_800_best_latest`` apart from ``gaucho_ellipse2d_dn``, and
started from the same checkpoint, so the denoising term is the only difference
between the two runs.

Denoising queries are noised copies of the annotations; they currently learn
box, centre, depth, rotation and size but never the ellipse, so the branch that
the reported metric actually reads gets no signal from the densest, cleanest
supervision in the model.
"""

_base_ = ["./train_800_best_latest.py"]

model = dict(bbox_head=dict(gaucho_ellipse2d_dn=True))
