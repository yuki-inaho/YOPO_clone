"""The depth anchor, end to end -- the residual is learned, in the model.

Three measurements set this up.  The 3D residual is entirely range: 21.9 mm
along the line of sight against 2.2 mm across it, on 18 mm objects.  Replacing
only the range with the annotation's multiplies shared AP20 by fifteen.  And
the depth channel the model already receives reads the object centre to
5.3 mm.  A post-hoc substitution of the range at inference reached AP20 0.3859
-- but that is a hand-written rule outside the model, not model performance,
and it cannot learn what a rule cannot express: occlusion, speckle, and the
surface-to-centre offset.

So the anchor lives in the forward pass instead.  At each decoder layer the
head samples the metric depth at the query's predicted centre (5x5 valid-pixel
median, image-median fallback where the sensor has no return) and *adds* it to
the regressed z, which turns the regression into a residual while everything
downstream -- losses, targets, denoising, matching, inference -- continues to
see absolute metres, unchanged.

The launch checkpoint has ``cop_z_out`` zeroed, so the run starts at exactly
"trust the sensor" (the substitution's operating point) and training can only
move away from it by reducing the loss.  Falling below AP20 0.3859 therefore
indicates a defect, not a preference; the ceiling to compare against is the
range oracle at 0.6796.
"""

_base_ = ["./train_1024_res.py"]

model = dict(
    bbox_head=dict(
        # Metres per un-normalized unit of the depth channel; measured on
        # 4,760 annotated centres (median reading error 5.32 mm at this value).
        sensor_depth_scale=3.90524303e-3,
        sensor_depth_anchor=True,
    ),
)

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
