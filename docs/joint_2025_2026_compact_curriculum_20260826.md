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
- Post-fix standalone validation completed all 511 images: AP50 `0.4420`,
  AP75 `0.1561`, AP50:95 `0.2054`. It emitted no 3D metric, as required by
  Stage 1.
- Exact resume confirmed `epoch=5, iter=425`; epoch 6 continued at base LR
  `1e-4` with the restored optimizer state and finite losses.
- First in-loop validation at epoch 10: AP50 `0.5653`, AP75 `0.2103`,
  AP50:95 `0.2656`. AP50:95 improved by `+0.0602` over the standalone epoch-5
  baseline. `best_AP50_95_epoch_10.pth`, `best_AP75_epoch_10.pth`, and the
  exactly resumable `epoch_10.pth` were written.

## Time-boxed production execution

The user requested completion in roughly one to two additional hours. The
configured upper bounds remain available for later saturation runs, while this
production pass used explicit CLI bounds of Stage 1/2/3/4 = `20/10/5/20`
epochs. Every transition still used a saved best checkpoint and the same
fail-closed metric gates.

### Stage 1: 2D detection

- Stop reason: requested time-box boundary after the epoch-20 validation.
- Epoch 15: AP50 `0.6165`, AP75 `0.2422`, AP50:95 `0.2970`.
- Epoch 20: AP50 `0.6272`, AP75 `0.2600`, AP50:95 `0.3079`.
- Accepted checkpoint: `best_AP50_95_epoch_20.pth`.

### Stage 2: parallel pose

- CLI bound: 10 epochs, initialized model-only from the accepted Stage-1
  checkpoint.
- Epoch 5: AP50:95 `0.3046`, exact 3D IoU@0.25 `0.0079`.
- Epoch 10: AP50 `0.6564`, AP75 `0.2581`, AP50:95 `0.3167`; exact 3D
  IoU@0.10/0.25/0.50/0.75 = `0.0256/0.0088/0.0005/0.0000`.
- 10-degree pose AP at 2/5/10/100 cm =
  `0.1631/0.1964/0.2169/0.2313`.
- The AP50:95 retention gate passed: Stage 2 exceeded Stage 1 by `+0.0088`.
- Accepted checkpoint: `best_3d_iou_0.25_epoch_10.pth`.

### Stage 3: auxiliary CoP and KFIoU

- CLI bound: 5 epochs, initialized model-only from the accepted Stage-2
  checkpoint.
- Physical batch 30 passed with a measured process allocation of 31,266 MiB.
- The newly enabled CoP size-chain loss fell from `15.8211` at the first
  logged update to `0.0026` at epoch 5; all losses remained finite.
- Epoch 5: AP50 `0.6852`, AP75 `0.2696`, AP50:95 `0.3307`; exact 3D
  IoU@0.10/0.25/0.50/0.75 = `0.0288/0.0107/0.0007/0.0000`.
- 10-degree pose AP at 2/5/10/100 cm =
  `0.1816/0.2197/0.2487/0.2609`.
- Accepted checkpoint: `best_3d_iou_0.25_epoch_5.pth`.

### Stage 4: CoP-primary GWD

- CLI bound: 20 epochs, initialized model-only from the accepted Stage-3
  checkpoint.
- Physical batch 30 and FP16 AMP passed the first optimizer update at about
  30,134 MiB process allocation, but failed deterministically after four full
  epochs: one to three query rows became non-finite before Hungarian
  assignment. The fail-closed check stopped both attempts with status 1.
- `epoch_4.pth` was scanned before reuse: all 1,033 model tensors and all 781
  optimizer-state entries were finite. The failure is therefore an FP16
  forward-range issue after additional GWD updates, not a corrupt source
  checkpoint.
- Stage 4 now uses BF16 AMP with loss scale `1.0` and recovery LR `5e-5`.
  BF16 retains tensor-core execution and the AMP memory class while providing
  FP32-like exponent range. The finite cumulative-epoch-5 checkpoint is used
  model-only and the remaining 15 epochs complete the requested 20-epoch
  Stage-4 budget.
- The installed MMCV CUDA extension has no native BF16 deformable-attention
  kernel. `AmpSafeMultiScaleDeformableAttention` therefore runs that extension
  alone in FP32 and casts its finite result back to BF16; all other eligible
  operations remain under BF16 autocast.
- Exact 3D evaluation now admits Gram error up to `5e-4` before SVD projection
  to SO(3). This covers the measured BF16 error (`1.498e-4`) while the existing
  non-uniform-scale test continues to reject genuinely anisotropic transforms.

#### Stage-4 results

- Cumulative epoch 10 (local BF16 epoch 5): AP50 `0.6929`, AP75 `0.2833`,
  AP50:95 `0.3389`; exact 3D IoU@0.10/0.25/0.50/0.75 =
  `0.0406/0.0166/0.0017/0.0000`.
- Cumulative epoch 20 (local BF16 epoch 15): AP50 `0.6907`, AP75 `0.2871`,
  AP50:95 `0.3396`; exact 3D IoU@0.10/0.25/0.50/0.75 =
  `0.0409/0.0163/0.0015/0.0000`.
- Final 10-degree pose AP at 2/5/10/100 cm =
  `0.1802/0.2234/0.2498/0.2627`.
- Selection: use `best_3d_iou_0.25_epoch_5.pth` for the 3D objective; use
  `best_AP50_95_epoch_15.pth` when 2D AP is primary. The small final IoU@0.25
  change (`-0.0003`) is why the metric-specific checkpoints are retained.
- Peak observed process allocation: 32,034 MiB of 32,607 MiB. Peak MMEngine
  tensor memory: 30,024 MiB. Physical batch remained 30 throughout.
- Stop reason: requested cumulative 20-epoch Stage-4 budget reached; final
  511-image validation completed and the process exited with status 0.
- Time from the user's one-to-two-hour boundary request to final validation:
  approximately 1 hour 29 minutes.

#### Checkpoint integrity

| Checkpoint | SHA-256 |
|---|---|
| Stage-1 2D best | `4fd2ed05dbc29a0e0c10b9a7ffd1a88758a307597d1ff89ec70f5d2b76edbfb3` |
| Stage-2 3D best | `c55768b936df169a5d9c06f0288e2abe12d1cc3a0c68eaac74c88c4b031f6bc6` |
| Stage-3 3D best | `b572dfd3b7536ed9432b26d0c1d64fd258c2e656cd4503d7a80db9b9102e8ee8` |
| Stage-4 3D best | `596046e443199b1fc749c5d085e38384dd7eb7900bb6d5aad18b55236036cc52` |
| Stage-4 2D best | `ff038bd9abc72d9f5177fc97b2f4a6e4239aaed42ccfd0042d74ca448e22d119` |
| Stage-4 final resumable epoch 15 | `92303695295e372b2d4e0fdf29b27d814cdc01b53f1434991326fb348b2d2d53` |

Focused regression evidence after the BF16/MMCV and exact-geometry fixes:
`148 passed`; Ruff and `git diff --check` passed.
