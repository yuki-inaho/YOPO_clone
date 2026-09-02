"""Coverage ceiling for the 512-query model, under the fair evaluation stack.

``q512`` was judged on mAP alone and recorded as no help.  mAP confounds
coverage with ranking, and coverage is what the 2D gap turned out to be: the
256-query model's queries never reach 22% of the objects, a ceiling below
rtmdet's achieved output.  This re-reads that run for the quantity that
actually binds.
"""

_base_ = ["./train_800_centre_q512.py"]
