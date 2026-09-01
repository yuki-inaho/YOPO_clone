"""The depth anchor, with denoising queries left out of it.

The anchored run reached a range error of 8.8 mm and shared AP20 0.21 -- a
4.7x gain over the unanchored 0.045 -- but then oscillated below the 0.386 that
the same anchor reaches as a fixed rule, which is where a zero-initialised
residual starts.  So the residual is being trained *away* from the sensor.

The candidate mechanism is the denoising stream.  Its queries are noised copies
of the annotations, so depth sampled at their displaced centre lands off the
object; ``dn_loss_z`` then asks the residual to undo a displacement the residual
cannot observe, and ``cop_z_out`` is shared with the matching queries, so that
correction leaks into them.  The measured ratio agrees: ``dn_loss_z / loss_z``
sits at 1.00 in the anchored run against 0.58-0.82 in the unanchored one.

This run anchors matching queries only; denoising keeps regressing absolute
depth exactly as before.  Everything else is identical, so the difference is
attributable to that split alone.
"""

_base_ = ["./train_1024_anchor.py"]

model = dict(bbox_head=dict(sensor_depth_anchor_denoising=False))
