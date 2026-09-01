"""Push input resolution again -- the one lever with measured evidence behind it.

Object size, not sensor resolution, is what this dataset is short of: 18.1% of
the annotated stems have a short side under 16 px at the native 736x512, and
every resize upward has paid for itself under the reference protocol --
640x445 -> 736x512 -> 800x557 moved rotated-NMS mAP50 from 0.6533 raw to
0.7113 and then 0.7312.  The frames are upsampled, so no new detail enters;
what changes is how many feature-map cells a small object covers.

1024x713 is 1.64x the pixels of 800x557.  At a fixed 24 GB that forces batch 20
down to 14, so this run is not a pure single-factor change -- resolution and
batch move together, which is stated rather than hidden.  Everything else,
including the ellipse-aware matching and ranking that won the previous
comparison, is held fixed.
"""

_base_ = ["./train_800_best_latest.py"]

scale_1024 = (1024, 768)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=scale_1024, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="RandomFlipFor9DPose", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(0.01, 0.01)),
    dict(type="Pack9DPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadRawDepthImageWithValidMask"),
    dict(type="Load9DPoseAnnotations", with_bbox=True, with_centers_2d=True,
         with_z=True, with_obb_gaussian=True),
    dict(type="ConcatRawDepthToImage", depth_scale=255.0),
    dict(type="ResizeforPose", scale=scale_1024, keep_ratio=True),
    dict(type="ResizeOBBGaussians"),
    dict(type="Pack9DPoseInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "intrinsic", "models_info_path")),
]

# 800x557 at batch 20 measured 18.9 GB allocated; 1.64x the pixels puts the
# marginal cost near 1.31 GB per sample, so 14 lands around 20.7 GB.
train_dataloader = dict(batch_size=14, num_workers=8,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=2, num_workers=4,
                      dataset=dict(pipeline=val_pipeline))

max_epochs = 16
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=2)

load_from = None
