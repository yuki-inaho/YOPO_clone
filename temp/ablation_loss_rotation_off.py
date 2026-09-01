"""One-factor ablation: does keeping rotation supervision during Stage B help?

Control is temp/train_gaucho_stageB_native.py, unchanged.  This treatment
changes exactly one thing: loss_rotation is zeroed and the rotation branch is
frozen, so no rotation gradient reaches the model during Stage B.

Deliberately NOT changed at the same time: centre/depth detach, projection
weight, learning rate, schedule, seed, batch size, resolution, evaluator.
Changing more than one would make the result uninterpretable.

Note this measures "continuing rotation supervision during Stage B", not
"a model that never had rotation pretraining" -- the Stage A checkpoint both
arms start from was itself trained with loss_rotation active.  Isolating the
pretraining would require restarting from a pose-free 2D checkpoint.
"""

_base_ = ["./train_gaucho_stageB_native.py"]

model = dict(bbox_head=dict(loss_rotation=dict(loss_weight=0.0)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.reg_rotation_branch": dict(lr_mult=0.0),
            "bbox_head.cop_rotation_net": dict(lr_mult=0.0),
            "bbox_head.cop_rotation_out": dict(lr_mult=0.0),
        }
    )
)

load_from = None
