# YOPO cu121 training (torch 2.4.0 + cu121)

This branch migrates YOPO training and inference onto a **uv-managed repo-local
venv** with **torch 2.4.0+cu121 / mmcv 2.2.0**. Verified end-to-end on an NVIDIA
L4 (sm_89): a 1-epoch smoke trains the real R50 DINO9DCenter2DPose model on
synthetic NOCS data and saves a checkpoint; a separate smoke loads the official
checkpoint and runs inference on GPU.

## Stack

- Python 3.10, **torch 2.4.0+cu121**, **torchvision 0.19.0+cu121**, `numpy<2`
- **mmcv 2.2.0** from the prebuilt cu121 manylinux wheel
  (`https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl`)
  — no source build; CUDA ops (nms, roi_align, …) compiled for cu121 at
  factory and run on sm_89 (L4).
- **mmengine 0.10.x** from PyPI.
- **yopo 3.3.0** — the repo root is made importable develop-style via a `.pth`
  file (`just sync` writes `_yopo_src.pth` into the venv `site-packages`).
  YOPO is a renamed MMDetection fork; it is **not** pip-installed. All internal
  imports use `from yopo`.
- `pyproject.toml` uses `[tool.uv] package = false`, pinned CUDA index for
  torch/torchvision (`[[tool.uv.index]]`), and `[tool.uv.sources]` for the
  prebuilt mmcv wheel URL.

## One-time setup

```bash
# Clone the repo and check out the cu121 branch
git clone git@github.com:yuki-inaho/YOPO.git
cd YOPO
git checkout cu121

# Full setup: uv sync + .pth (run once)
just setup

# Verify the environment
just env-doctor
```

`just env-doctor` should print:
```
torch 2.4.0+cu121 | cuda 12.1
torchvision 0.19.0+cu121
mmcv 2.2.0
mmengine 0.10.7
yopo 3.3.0
cuda_available True
device NVIDIA L4
```

`just sync` (called by `just setup`) does:
1. `uv sync` — provisions torch 2.4.0+cu121, the mmcv 2.2.0 cu121 wheel,
   numpy<2, and all runtime deps from `pyproject.toml`.
2. Writes `_yopo_src.pth` into `.venv/lib/python3.10/site-packages/` so
   `import yopo` resolves to the repo root.

## Synthetic data & checkpoint (smoke prerequisites)

```bash
# Generate a 4-frame synthetic NOCS-format dataset
just gen-synthetic          # → data/nocs_smoke/

# Download the official YOPO R50 checkpoint (~196 MB)
just download-ckpt          # → checkpoints/nocs_yopo_real_camera_r50.pth
```

The synthetic data generator (`scripts/gen_synthetic_nocs.py`) creates the full
NOCS directory layout under `data/nocs_smoke/` with `real/`, `camera/`,
`camera_full_depths/`, and `segmentation_results/` subdirectories. Each frame
has colour, depth (uint16), and label pkls matching the keys expected by
`NOCSDataset` and its transforms.

## GPU smoke training (1 epoch)

```bash
just smoke-train           # → work_dirs/smoke_train/epoch_1.pth
```

What it does:
- Uses `tools/train.py` with the smoke config `temp/smoke_nocs_r50_1iter.py`.
- The smoke config inherits from `configs/yopo/nocs_yopo_real_camera_r50.py`
  and overrides: `data_root=data/nocs_smoke`, `max_epochs=1`, `batch_size=2`,
  `num_workers=0`, `load_from=None`, val/test all disabled (`val_dataloader`,
  `val_cfg`, `val_evaluator`, `test_dataloader`, `test_cfg`, `test_evaluator`
  all set to `None`), checkpoint saved at interval 1.
- Backbone weights are fetched from `torchvision://resnet50` (small download,
  cached after first run).
- Expected output: 4 iterations of loss values in the log, then
  `Saving checkpoint at 1 epochs`, exit 0.

## GPU smoke inference

```bash
just smoke-infer           # → "SMOKE INFER OK"
```

What it does:
- `scripts/smoke_infer.py` loads the config, builds the DINO9DCenter2DPose
  model via `MODELS.build()` (with `init_default_scope('yopo')` set),
  loads the official checkpoint, moves to GPU, loads a single synthetic
  test frame via the test pipeline, runs `model.test_step()`, and prints
  prediction shapes/keys.
- Expected output: `SMOKE INFER OK`, exit 0.

## Full training / evaluation (real data)

Once the real NOCS or HouseCat6D dataset is placed under `data/`:

```bash
# Training
just train configs/yopo/nocs_yopo_real_camera_r50.py

# Evaluation
just test configs/yopo/nocs_yopo_real_camera_r50.py checkpoints/nocs_yopo_real_camera_r50.pth
```

## Known gotchas

- **val/test all-None constraint**: mmengine requires `val_dataloader`, `val_cfg`,
  and `val_evaluator` to be either all `None` or all non-`None`. Same for
  `test_dataloader`/`test_cfg`/`test_evaluator`. If disabling validation for a
  smoke config, set all six to `None` (see `temp/smoke_nocs_r50_1iter.py`).
- **`init_default_scope('yopo')`**: When building a model outside a Runner
  (e.g. in a standalone inference script), you must call
  `from mmengine.registry import init_default_scope; init_default_scope('yopo')`
  before `MODELS.build()`, otherwise `DetDataPreprocessor is not in the registry`.
- **`batch_size`**: `batch_size=1` can break BatchNorm layers in the backbone
  or transformer heads. The smoke config uses `batch_size=2`.
- **Synthetic label keys**: The label pkl must include all keys expected by
  `NOCSDataset._parse_instance_info` and the corresponding transforms. For
  train: `class_ids`, `instance_ids`, `bboxes`, `translations`, `rotations`,
  `sizes`, `scales`. For test/val: `gt_class_ids`, `gt_bboxes`, `gt_RTs`,
  `gt_scales`, `gt_handle_visibility`.
- **Intrinsics**: Must match the hardcoded values in
  `yopo/datasets/pose_estimation/nocs_dataset.py` `SPLIT_INFO`:
  `[591.0125, 590.16775, 322.525, 244.11084]` for real, `[577.5, 577.5, 319.5, 239.5]` for camera.
- **Depth images**: Even though the R50 config pipeline does **not** include
  `LoadDepthImageFromFile`, the dataset still requires depth files to exist
  on disk (checked in `parse_data_info`).
- **Prebuilt mmcv wheel**: The cu121 wheel is manylinux; it requires glibc ≥ 2.17.
  It ships compiled CUDA 12.1 kernels. The L4 driver (CUDA 12.7) is backward
  compatible with cu121 runtime via CUDA minor-version compat.

## Non-goals (not covered here)

- Real NOCS/HouseCat6D dataset preparation and full training to paper accuracy.
- Multi-GPU distributed training (`tools/dist_train.sh`).
- ONNX / mmdeploy export.
- Swin-L / HouseCat6D config verification (only R50 / NOCS).

## File reference

| File | Purpose |
|:---|:---|
| `pyproject.toml` | uv/cu121 dependency stack |
| `justfile` | Recipe shortcuts (`sync`, `setup`, `env-doctor`, `gen-synthetic`, `download-ckpt`, `smoke-train`, `smoke-infer`, `train`, `test`) |
| `scripts/gen_synthetic_nocs.py` | Synthetic NOCS dataset generator |
| `temp/smoke_nocs_r50_1iter.py` | 1-epoch smoke training config |
| `scripts/smoke_infer.py` | Standalone inference smoke |
| `configs/yopo/nocs_yopo_real_camera_r50.py` | R50 NOCS training config |
| `checkpoints/nocs_yopo_real_camera_r50.pth` | Official R50 checkpoint (~196 MB) |
| `data/nocs_smoke/` | Generated synthetic dataset |
| `work_dirs/smoke_train/` | Smoke training output (checkpoint + logs) |
