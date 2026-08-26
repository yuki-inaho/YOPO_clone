"""RGB-D 3D BBOX training on the post-hoc 736x512 2026 dataset."""

_base_ = ["./nocs_fruit_obb_rgbd_2026_800x600_3dbbox.py"]

data_root = "data/fruit_obb_rgbd_train1345-test181_20260825_yopo_3dobb_736x512/"

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
