"""RGB-D 3D BBOX training on the portable 2026 DOTA conversion."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_transfer.py"]

# Every generated label carries its own intrinsic, image_size_wh and explicit
# OBB coordinate scale. The inherited split constants are legacy fallbacks.
data_root = "data/fruit_obb_rgbd_train1345-test181_20260825_yopo_3dobb_800x600/"

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
