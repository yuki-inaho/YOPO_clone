"""Teacher-free inference for the weak-DA corrected tomato OBB detector."""

_base_ = [
    "./rotated_deformable_detr_tomato_obb_corrected_da_weak_gwd_stage2.py"
]

load_from = None
resume = False
