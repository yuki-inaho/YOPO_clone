"""NMS-free v2: one-to-one inference with a training-only one-to-many head."""

_base_ = ["./nocs_custom_fruit_rgbd_nmsfree_control_continue5.py"]

# The main/inference head keeps the accepted binary Focal supervision.  The
# separate O2M classifier/regressor supervises shared decoder representations
# during training and is absent from bbox_head.forward()/predict().
model = dict(
    bbox_head=dict(
        o2m_aux_topk=2,
        o2m_aux_loss_weight=0.1,
    ),
)
