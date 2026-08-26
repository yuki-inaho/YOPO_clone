"""RGB-D 3D BBOX training on the portable Jun30-2025 DOTA conversion."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_transfer.py"]

# Labels in this dataset carry per-frame intrinsics, image_size_wh, and an
# explicit OBB coordinate scale. The inherited intrinsic/0.8 values remain
# compatibility fallbacks for legacy labels and are not applied here.
data_root = "data/fruits_detection_Jun30-2025_stem_rgbd_736x512/"

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
