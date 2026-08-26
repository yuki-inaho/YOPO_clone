"""Phase 1 with the established weak tomato augmentation policy.

The policy mirrors rotated_rtmdet_jax/tomato_jun30_mmrotate_weak:
vertical flip p=0.75 followed by unconditional YOLOX HSV jitter. Validation
remains deterministic and augmentation-free.
"""

_base_ = ["./rotated_deformable_detr_tomato_obb_corrected_riou_stage1.py"]

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, box_type="rbox"),
    dict(type="Resize", scale=(800, 600), keep_ratio=False),
    dict(type="RandomFlip", prob=0.75, direction="vertical"),
    dict(
        type="YOLOXHSVRandomAug",
        hue_delta=5,
        saturation_delta=30,
        value_delta=30,
    ),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

