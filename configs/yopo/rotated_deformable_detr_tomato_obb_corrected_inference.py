"""Teacher-free inference config for either corrected-data OBB stage."""

_base_ = ["./rotated_deformable_detr_tomato_obb_corrected_gwd_stage2.py"]

load_from = None
resume = False
