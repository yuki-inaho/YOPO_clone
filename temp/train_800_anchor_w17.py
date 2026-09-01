"""Widen the depth sampling window, because the tail is what costs.

Switching z from L2 to L1 moved shared AP20 from 0.258 to 0.415, which says the
anchor's error distribution -- not its median -- is what the model was fighting.
So the window that produces that distribution is worth measuring rather than
assuming, and a sweep at both resolutions says:

    window   p50    p75    p90
      5x5   5.79  11.91  44.89     (current)
     11x11  5.27  10.98  29.37
     17x17  5.07  10.41  25.46

Median aggregation throughout: taking the minimum instead, to favour the nearest
unoccluded surface, is far worse in the tail (p90 338 mm at 5x5).  A 17x17
window is still well inside these objects, whose boxes are 36 px wide at this
resolution, so it is not reaching for background.

Config-only change; the window was already a parameter.
"""

_base_ = ["./train_800_anchor_l1.py"]

model = dict(bbox_head=dict(sensor_depth_window=8))
