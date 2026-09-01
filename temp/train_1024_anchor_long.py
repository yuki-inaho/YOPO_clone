"""The anchored model, trained long enough for the residual to converge.

The 16-epoch anchored run reached shared AP20 0.2581 and a range error of
8.10 mm -- 5.7x the unanchored 0.0451 and 21.9 mm -- but its best was at epoch
10 and it was still moving, and the residual it is learning starts from zero by
construction.  The fixed-rule operating point is 0.3859 and the range oracle is
0.6796, so there is a lot of room left and no sign the trajectory had settled.

Excluding denoising queries from the anchor was tried and is refuted: with the
anchor on matching queries only, ``cop_z_out`` has to emit a residual for one
stream and an absolute depth for the other, they differ by about 0.4 m, and one
linear head cannot do both -- range error went to 116 mm and AP20 to zero.  So
the anchor stays on every query, exactly as in the run this extends.

Nothing changes but the number of epochs.
"""

_base_ = ["./train_1024_anchor.py"]

max_epochs = 40
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)
