"""Five-epoch continuation control from the verified Q150 Stage 4 best."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage4_full.py"]

load_from = (
    "work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_from_2d_stage4_full/"
    "best_3d_iou_0.50_epoch_5.pth"
)

# This control changes neither the model objective nor optimizer. It isolates
# the gain from five more Stage-4 epochs when comparing projection GWD.
model = dict(bbox_head=dict(loss_projection=None))
