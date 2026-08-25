"""Recover the full 3D Stage-4 objectives from the 2D OBB foundation."""

_base_ = [
    "./nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1.py"
]

# The twenty-epoch foundation establishes the query-level 2D detector and OBB
# representation.  Restore the ordinary Stage-4 z/size/rotation objectives and
# their frozen teacher, while retaining OBB supervision at weight 1.  Direct
# projection was already rejected in isolation, so it remains disabled here.
load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_2d_obb_foundation_full/epoch_20.pth"
)

model = dict(
    bbox_head=dict(
        loss_projection=None,
    ),
)
