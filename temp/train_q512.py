"""Raise the query budget from 256 to 512.

The largest annotated frame in this data holds 205 objects, so 256 queries are
nominally sufficient and this is not a capacity fix in the counting sense.  It
is a redundancy change: a one-to-one matcher gives each object exactly one
positive, and with dense, near-identical fruit the assignment is easy to get
locally wrong.  More candidates give the matcher more chances to place a good
query on a small object without displacing a neighbour.

Cheap to test, and it is the step the review put before any architectural
change (Q1000 measured 0.7277 on the reference side, against 0.8136 for the
dense detector on the same split -- candidate count alone does not close it).
"""

_base_ = ["./gaucho_800x600_base.py"]

max_objects = 512

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        test_cfg=dict(max_per_img=max_objects),
        loss_ellipsoid=dict(include_center=False),
    ),
)

# Queries are the dominant term in decoder attention memory, so the batch comes
# down with them.  Batch 24 at 256 queries hit a CUDA OOM at epoch 9 -- the
# allocation itself fitted but ~1 GB of reserved-but-unallocated fragmentation
# did not -- so this leaves real headroom rather than the last megabyte.
train_dataloader = dict(batch_size=12)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
