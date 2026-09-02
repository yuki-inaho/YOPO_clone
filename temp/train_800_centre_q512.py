"""Combine everything that raised the 2D ceiling, and see if the score follows.

Two changes have moved the geometry oracle -- the number that says what this
detector could reach with perfect ellipses:

    baseline           ceiling 0.8215   actual 0.7312
    + 512 queries              0.8294          0.7232
    + centre term              0.8273          0.7365

Queries raised the ceiling but not the score; the centre term raised both.  They
act on different things -- recall and duplication on one side, geometry on the
other -- so this runs them together, on top of the depth anchor and its window.

Batch comes down to 12 with the query count, as it did in the queries-only run.
"""

_base_ = ["./train_800_centre.py"]

max_objects = 512

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(test_cfg=dict(max_per_img=max_objects)),
)

train_dataloader = dict(batch_size=12)
