# YOPO: You Only Pose Once

Official implementation of **YOPO** from the paper:

**You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation**  
[Hakjin Lee](https://scholar.google.com/citations?user=GIdEXLsy77UC), [Junghoon Seo](https://mikigom.github.io/), [Jaehoon Sim](https://www.linkedin.com/in/jaehoon-sim-801404175/)

[\[arXiv\]](https://arxiv.org/abs/2508.14965) 
[\[Project page\]](https://mikigom.github.io/YOPO-project-page/)

This paper is accepted to **IEEE ICRA 2026**.

## Overview

YOPO is a single-stage transformer framework for RGB-only category-level 9D pose estimation.  
This repository is built on MMDetection and contains YOPO-specific configurations and model extensions.

Current YOPO entry points are in:

- `configs/yopo/`
- `yopo/models/detectors/sixd_pose/`
- `yopo/models/dense_pose_heads/`
- `yopo/datasets/pose_estimation/`
- `yopo/evaluation/metrics/`

## Model Zoo
### NOCS (REAL275)

| Model | Config | IoU50 | IoU75 | 10deg10cm | Checkpoint |
|---|---|---:|---:|---:|---|
| YOPO R50 | [file](configs/yopo/nocs_yopo_real_camera_r50.py) | 67.1 | 16.6 | 40.7 | [link](https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/nocs_yopo_real_camera_r50.pth) |
| YOPO Swin-L | [file](configs/yopo/nocs_yopo_real_camera_swinl.py) | 71.6 | 16.4 | 52.8 | [link](https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/nocs_yopo_real_camera_swinl.pth) |
| YOPO Swin-L* | [file](configs/yopo/nocs_yopo_real_camera_swinl_finetune_real.py) | 79.6 | 19.6 | 54.1 | [link](https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/nocs_yopo_real_camera_swinl_finetune_real.pth) |

### HouseCat6D

| Model | Config | IoU25 | IoU50 | 10deg10cm | Checkpoint |
|---|---|---:|---:|---:|---|
| YOPO Swin-L | [file](configs/yopo/housecat6d_yopo_swinl.py) | 71.3 | 34.8 | 33.3 | [link](https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/housecat6d_yopo_swinl.pth) |

### Fruit RGB-D (custom validation split)

| Model | HBB AP50 | Ellipse AP50 | Projection AP50 | Shared AP25 | Strict 3D IoU25 | Best-only bundle |
|---|---:|---:|---:|---:|---:|---|
| YOPO YOLO26m RGB-D | 0.8622 | 0.8614 | 0.8342 | 0.2679 | 0.1915 | [tar.zst](https://github.com/yuki-inaho/YOPO_clone/releases/download/yolo26m-rgbd-20260906/yopo_yolo26m_rgbd_best_20260906.tar.zst) |

## Quickstart (uv + RTX 5090 / cu128)

The `rgb-d` branch uses **uv** with a repo-local Python 3.10 venv, PyTorch
2.8.0 + CUDA 12.8, and the prebuilt `sm_120` MMCV CUDA-ops wheel for RTX 5090.
No Docker or local MMCV source build is required.

```bash
# Clone and switch to the RGB-D branch
git clone https://github.com/yuki-inaho/YOPO_clone.git
cd YOPO_clone
git checkout rgb-d

# One-time setup
just setup

# Verify the environment
just env-doctor
# Expected: torch 2.8.0+cu128 / mmcv 2.2.0 / mmengine 0.10.7 /
#           cuda_available True / NVIDIA GeForce RTX 5090

# Generate synthetic data for smoke tests
just gen-synthetic

# Download the official R50 checkpoint
just download-ckpt

# GPU smoke training (1 epoch, saves checkpoint)
just smoke-train

# GPU smoke inference (loads official checkpoint, runs forward pass)
just smoke-infer
```

See [docs/CU128_RTX5090.md](docs/CU128_RTX5090.md) for environment details.
The legacy L4/cu121 smoke-test guide remains at
[docs/CU121_TRAINING.md](docs/CU121_TRAINING.md).

## Environment (Docker)

The target base environment is:

- Docker image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

## Installation

```bash
# 1) clone
cd /path/to/YOPO

# 2) install OpenMMLab core deps with openmim
apt update -y
apt-get install libxcb1 ffmpeg libsm6 libxext6 -y
pip install -U pip setuptools wheel openmim mmengine
mim install mmcv==2.2.0

# 3) install this repo (setup.py)
pip install -v -e .

# 4) (optional) install extra deps listed by the repo
#    use this if you need optional scripts/utilities beyond core train/test
pip install -r requirements.txt
```

## Datasets

If you prepared data following AG-Pose, keep AG-Pose preprocessing outputs but map them to this repository layout.

- AG-Pose reference: https://github.com/Leeiieeo/AG-Pose
- Use the dataset processing instructions and download links from AG-Pose README (`Data Processing` section), then place outputs under `data/`.

### NOCS (for `NOCSDataset`)
Expected root: `data/nocs/`

Required structure for this codebase:

```text
data/nocs/
  camera/
    train_list.txt
    val_list.txt
    <scene>/<frame>_color.png
    <scene>/<frame>_depth.png
    <scene>/<frame>_label.pkl
  camera_full_depths/
    <scene>/<frame>_composed.png
  real/
    train_list.txt
    test_list.txt
    <scene>/<frame>_color.png
    <scene>/<frame>_depth.png
    <scene>/<frame>_label.pkl
  segmentation_results/
    CAMERA25/
      results_val_<scene>_<frame>.pkl
    REAL275/
      results_test_<scene>_<frame>.pkl
```

Notes:

- The list files (`train_list.txt`, `test_list.txt`, etc.) should store frame IDs without suffixes (the loader appends `_color.png`, `_depth.png`, `_label.pkl`).
- `real_test` evaluation uses `segmentation_results/REAL275`.
- For `camera` samples, the loader looks for composed depth maps in `camera_full_depths`.

### HouseCat6D (for `HouseCat6DDataset`)
Expected root: `data/housecat6d/`

Required structure:

```text
data/housecat6d/
  scene*/            # train
  test_scene*/       # test
```

Each scene should include at least:

- `rgb/*.png`
- `depth/*.png`
- `labels/*_label.pkl`
- `intrinsics.txt`


## Training

### NOCS (R50)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh configs/yopo/nocs_yopo_real_camera_r50.py 4 --auto-scale-lr
```

### NOCS (Swin-L)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh configs/yopo/nocs_yopo_real_camera_swinl.py 4 --auto-scale-lr
```

### HouseCat6D (Swin-L)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh configs/yopo/housecat6d_yopo_swinl.py 4 --auto-scale-lr
```

## Fine-tuning (NOCS Real)

The repository includes a real-data fine-tuning setup:

- Config: `configs/yopo/nocs_yopo_real_camera_swinl_finetune_real.py`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh configs/yopo/nocs_yopo_real_camera_swinl_finetune_real.py 4 --auto-scale-lr \
  --cfg-options load_from=work_dirs/nocs_yopo_real_camera_swinl/epoch_12.pth
```

## Fruit RGB-D YOLO26m production model

The selected model uses the accepted DEIMv2 JAX model's official-pretrained
YOLO26m backbone layers 0-10 as the YOPO RGB branch. It retains the existing
HGNetV2-B0 depth branch, residual fusion, PortableHybridEncoder, and YOPO query
decoder/2D/3D heads. It does **not** use the native YOLO Detect/OBB head or the
YOLO head-side PAN/FPN.

The run used batch 24, BF16, AMUSE, validation every five epochs, and early
stopping on shared AP25. Epoch 10 was selected; training stopped normally at
epoch 30 after four non-improving validation records. Independent evaluation
on 181 images reproduced HBB AP50 `0.8622128367`, ellipse AP50 `0.8613967896`,
projection AP50 `0.8342097371`, shared AP25 `0.2679015474`, and strict 3D IoU25
`0.1914560553`, with zero invalid projection/ellipsoid predictions.

The archive contains exactly one model checkpoint plus its resolved config,
metrics, manifest, public work record, and design. It excludes datasets,
optimizer state, raw logs, and intermediate or rejected checkpoints.

```bash
gh release download yolo26m-rgbd-20260906 \
  --repo yuki-inaho/YOPO_clone \
  --pattern 'yopo_yolo26m_rgbd_best_20260906.tar.zst'
sha256sum yopo_yolo26m_rgbd_best_20260906.tar.zst
# Expected: 31fe6dcfae643d87b542bd438cc9a63dc1b63f24f7616cca2b46a3dc41e00f90
tar --zstd -xf yopo_yolo26m_rgbd_best_20260906.tar.zst
sha256sum --check \
  yopo_yolo26m_rgbd_best_20260906/MANIFEST.sha256
```

The model-only checkpoint SHA256 is
`831c82632adf1beff15327f0f633570c08cf18d4b3e06660b97792b5157b4ad0`.
See the [work record](diary/workdoc_2026-09-06_yolo26m_rgbd_training.md) and
[transfer design](diary/design_yolo26m_rgbd_transfer.md) for the strict
250-leaf transfer, train-only boundary calibration, commands, and full metrics.

Evaluate after extracting the archive:

```bash
YOPO_BEST="$PWD/yopo_yolo26m_rgbd_best_20260906/yopo_yolo26m_rgbd_best_epoch10.pth"
uv run python tools/test.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage2_calibrated_full.py \
  "$YOPO_BEST"
```

### Legacy stage selection and reverse-KLD smoke

Before the YOLO26m transfer, the selected RGB-D baseline was stage 8:
`configs/yopo/nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py`. Stage 10 is
an experimental follow-up that changes only the direct ellipsoid KLD direction to
prediction-to-target and caps training at 15 epochs:
`configs/yopo/nocs_fruits_2026_rgbd_shared_stage10_reverse_kld.py`.

Run the two-update B24 capacity smoke from a stage-8 best checkpoint as a weights-only
initialization. `resume=False` intentionally creates a fresh optimizer; the inherited
training settings keep AMUSE, BF16, all loss families, and ellipsoid center learning.

```bash
: "${YOPO_DATA_ROOT:?set YOPO_DATA_ROOT to the preprocessed RGB-D dataset}"
: "${YOPO_STAGE8_BEST:?set YOPO_STAGE8_BEST to the stage-8 best checkpoint}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python tools/train.py \
  configs/yopo/nocs_fruits_2026_rgbd_shared_stage10_reverse_kld_capacity_smoke.py \
  --work-dir work_dirs/stage10_reverse_kld_capacity_smoke \
  --cfg-options load_from="$YOPO_STAGE8_BEST" resume=False \
    train_dataloader.dataset.data_root="$YOPO_DATA_ROOT/" \
    val_dataloader.dataset.data_root="$YOPO_DATA_ROOT/"
```

Do not promote a reverse-KLD checkpoint on one metric alone. The measured follow-up was
rejected because no single validation epoch simultaneously met shared AP25 `>=0.2218`,
strict 3D IoU25 `>=0.1960`, and projection AP50 `>=0.8388` with the 2D guards. Keep the
revalidated stage-8 best in that case. The rejected stage-9 `include_center=False` and
ellipsoid-only freeze are not part of stage 10.

## Evaluation

```bash
python tools/test.py <CONFIG> <CHECKPOINT>
```

Examples:

```bash
python tools/test.py configs/yopo/nocs_yopo_real_camera_r50.py <path_to_ckpt>
python tools/test.py configs/yopo/nocs_yopo_real_camera_swinl.py <path_to_ckpt>
python tools/test.py configs/yopo/housecat6d_yopo_swinl.py <path_to_ckpt>
```


## Citation

```bibtex
@inproceedings{lee2026yopo,
  title     = {You Only Pose Once: A Minimalist’s Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation},
  author    = {Lee, Hakjin and Seo, Junghoon and Sim, Jaehoon},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale)

This repository is based on MMDetection:

- https://github.com/open-mmlab/mmdetection
