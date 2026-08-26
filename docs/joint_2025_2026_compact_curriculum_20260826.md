# 2025+2026 compact RGB-D FULL curriculum (2026-08-26)

## Goal and fixed contracts

- Data: `data/fruits_rgbd_2025_2026_800x600_preprocessed/{2025,2026}`.
- Input: stored `4x600x800`; no runtime resize; RGB and mapped depth stay aligned.
- Model: RGB B1 + Depth B0, encoder/decoder 4 layers, FFN 1024,
  `d_model=256`, 256 queries.
- Runtime: RTX 5090 32 GB, physical batch 30, FP16 AMP loss scale 0.25.
- Optimizer: `AdamWScheduleFreeOptimizer`, base LR `1e-4`.
- Assignment: all stages retain classification + HBB L1 + GIoU Hungarian
  costs. Pose is never allowed to change object/query identity assignment.
- Initialization: production 512-sample Group Fisher transplant from the
  stopped Stage-12 epoch-20 model. It is a weight load, never an optimizer
  resume.

## Minimal four-stage curriculum

| Stage | Primary learning target | Epoch bound | Validation / transition |
|---|---|---:|---|
| 1 | HBB class/L1/GIoU + center2D only | 50 | best AP50:95; plateau patience 4 |
| 2 | independent parallel z/size/rotation | 20 | best exact 3D IoU@0.25; preserve 2D AP |
| 3 | parallel pose + auxiliary CoP, 2D OBB KFIoU | 20 | improve exact 3D IoU; no AP collapse |
| 4 | CoP primary, OBB GWD + projected-ellipsoid GWD | 100 | exact 3D IoU plateau patience 6 |

Stage 1 sets every metric-pose loss and pose-branch LR to zero. Its evaluator
uses `compute_pose_metrics=False`, so validation cannot spend time on or select
by untrained pose output. Stage 2 restores the transplanted parallel pose
heads. Stage 3 gives half of the historical pose-loss weight to each of the
parallel and auxiliary CoP paths, keeping the total budget unchanged. Stage 4
makes the trained CoP path primary and switches the OBB curriculum from the
IoU-like KFIoU objective to GWD plus projection consistency.

## Configs

1. `configs/yopo/nocs_fruits_2025_2026_rgbd_compact_curriculum_stage1_2d_full.py`
2. `configs/yopo/nocs_fruits_2025_2026_rgbd_compact_curriculum_stage2_parallel_pose.py`
3. `configs/yopo/nocs_fruits_2025_2026_rgbd_compact_curriculum_stage3_cop_aux_kfiou.py`
4. `configs/yopo/nocs_fruits_2025_2026_rgbd_compact_curriculum_stage4_cop_chain_gwd_full.py`

Every stage uses `load_from=None` and `resume=False` in source. Supply the
accepted predecessor through CLI. Only an interrupted run with the exact same
stage config may use `--resume` and its periodic checkpoint with optimizer
state.

## Transition gates

- Stage 1 -> 2: use `best_AP50_95_*.pth`; AP50:95 and AP75 must be finite.
- Stage 2 -> 3: use `best_3d_iou_0.25_*.pth`; AP50:95 may not be more than
  0.01 below Stage 1 best. Otherwise stay in Stage 2 and repair detection.
- Stage 3 -> 4: CoP auxiliary losses, KFIoU loss, gradients and all metrics
  must be finite; AP50:95 may not drop by more than 0.01 from Stage 2.
- Final: report AP50:95, AP75, exact 3D IoU@0.25/0.50/0.75, pose AP, peak
  VRAM, checkpoint SHA-256 and stop reason. A stage timeout or NaN is a failed
  gate, not permission to advance.

## Stage 1 launch

```bash
uv run python tools/train.py \
  configs/yopo/nocs_fruits_2025_2026_rgbd_compact_curriculum_stage1_2d_full.py \
  --work-dir /home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/compact_joint800_curriculum_stage1_2d_b30 \
  --cfg-options \
  load_from=/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/compact_b1b0_e4d4_ffn1024_800x600/stage12e20_group_fisher512_partial.pth
```

This command starts only after the implementation/config commit is pushed.
The first finite optimizer update and observed GPU peak are recorded before
the run is treated as healthy; the first AP50:95 validation is the first
accuracy gate.

## Run record

- Implementation/config commit: `4695d13` (pushed before training).
- First launch: `20260826_153828`, batch 30, base LR `1e-4`.
- Epoch 5 train result: total loss `15.9860`, bbox L1 `0.1031`, GIoU
  `0.6883`, center2D `0.0492`; all pose losses exactly zero and all logged
  values finite.
- Peak observed process GPU allocation: approximately 29.9 GiB; MMEngine
  peak tensor memory 27,651 MiB.
- The first validation correctly exposed a native-input metadata omission:
  prediction rescaling requires `scale_factor`, but the resize-free joint
  pipeline had not recorded identity scale. Training was stopped after the
  valid `epoch_5.pth` save.
- Fix: insert `AssertIdentityImageGeometry(image_size=(800, 600), channels=4)`
  in every joint train/validation pipeline and explicitly import it. Real
  samples from both years now verify as `4x600x800`, `img_shape==ori_shape`,
  and `scale_factor==(1.0, 1.0)`. Resume must use `epoch_5.pth` with optimizer
  state rather than restarting or loading it as model-only.
