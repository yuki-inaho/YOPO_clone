_base_ = ['./test_nocs_custom_rgbd_deim_cop.py']

# Compare baseline (no CoP) checkpoint for diagnosis: disable cop_chain in
# the head; everything else identical.
model = dict(bbox_head=dict(use_cop_chain=False))
