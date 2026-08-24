_base_ = ['./nocs_custom_real_hgnetv2_rgbd_deim_cop.py']

# Retrain from scratch with the fixed depth normalization (the old checkpoint
# is incompatible because the stem's input distribution changed).
load_from = None
resume = False
