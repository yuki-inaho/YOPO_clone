"""Q150 CoP curriculum initialized from the verified 2D foundation."""

_base_ = ["./nocs_custom_fruit_rgbd_3dbbox_cop_curriculum_base.py"]

max_objects = 150

# This checkpoint has the trained Q150 detector/center branches. Its parallel
# z/size/rotation branches are tensor-identical to the original epoch-10 pose
# checkpoint, so it is also the coherent frozen teacher for this curriculum.
foundation_checkpoint = (
    "work_dirs/nocs_custom_fruit_rgbd_2d_foundation_full/"
    "best_AP50_epoch_50.pth"
)
pose_teacher_checkpoint = foundation_checkpoint

model = dict(
    num_queries=max_objects,
    test_cfg=dict(max_per_img=max_objects),
    bbox_head=dict(
        test_cfg=dict(max_per_img=max_objects),
        pose_teacher_checkpoint=pose_teacher_checkpoint,
    ),
)

# Q150 plus dense depth context is heavier than the Q150 2D run and the old
# Q100 depth stage. Batch 22 was verified at 29.8 GiB process VRAM on the
# 32 GiB RTX 5090, retaining about 2.3 GiB for runtime variance.
train_dataloader = dict(batch_size=22)

# Preserve the converged detector with a conservative base LR, while giving
# the previously unused CoP predictors/fusions and depth query sampler enough
# learning rate to make each five-epoch PDCA stage meaningful.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.cop_": dict(lr_mult=50.0),
            "bbox_head.depth_query_sampler": dict(lr_mult=50.0),
        },
    ),
)
