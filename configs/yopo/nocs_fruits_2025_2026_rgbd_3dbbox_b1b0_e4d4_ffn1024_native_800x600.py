"""Compact 800x600 joint RGB-D model for architecture-first fine-tuning.

This keeps the 256-dimensional DINO/CoP interface and Q256 contract while
reducing the RGB backbone, transformer depth, and FFN hidden width.  A partial
checkpoint produced by ``tools/model_converters/transplant_compact_yopo.py``
can be supplied through ``--cfg-options load_from=...``.
"""

_base_ = [
    "./nocs_fruits_2025_2026_rgbd_3dbbox_"
    "stage10_mal_native_800x600.py"
]

rgb_b1_pretrained = (
    "https://github.com/Peterande/storage/releases/download/dfinev1.0/"
    "PPHGNetV2_B1_stage1.pth"
)

model = dict(
    backbone=dict(
        rgb_backbone=dict(
            name="B1",
            init_cfg=dict(
                _delete_=True,
                type="Pretrained",
                checkpoint=rgb_b1_pretrained,
            ),
        ),
        # The compact partial checkpoint carries the complete trained B0
        # branch, so this experiment has no work-dir-relative MAE dependency.
        depth_backbone=dict(init_cfg=None),
    ),
    neck=dict(in_channels=[256, 512, 1024]),
    encoder=dict(
        num_layers=4,
        layer_cfg=dict(
            ffn_cfg=dict(feedforward_channels=1024),
        ),
    ),
    decoder=dict(
        num_layers=4,
        layer_cfg=dict(
            ffn_cfg=dict(feedforward_channels=1024),
        ),
    ),
)

# This is a new topology, not an optimizer-state resume.  Pass the generated
# weight-only partial checkpoint as ``load_from`` at launch.
load_from = None
resume = False
