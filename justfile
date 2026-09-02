# ============================================================================
# YOPO — uv / RTX 5090 cu128 (torch 2.8.0 + cu128, mmcv 2.2.0) workflow.
#
# End goal: category-level 9D pose training AND inference progress on an RTX
# 5090 under a repo-local uv venv. The active project uses cu128 Torch and the
# prebuilt sm_120 MMCV wheel, numpy<2, and the `yopo` package made
# importable develop-style via a .pth (it is a renamed MMDetection fork, not a
# pip-installed package).
# ============================================================================

VENV_PATH := ".venv"
PY_VER := "python3.10"
PYTHON_EXEC := invocation_directory() + "/" + VENV_PATH + "/bin/python"
SITE_PACKAGES := VENV_PATH + "/lib/" + PY_VER + "/site-packages"

SMOKE_DATA := "data/nocs_smoke"
SMOKE_TRAIN_CONFIG := "temp/smoke_nocs_r50_1iter.py"

# Show the documented recipe list.
default:
    @just --list

list:
    @just --list

# Helper: ensure the repo-local venv has a usable Python.
check-venv:
    @if [ ! -x "{{ PYTHON_EXEC }}" ]; then \
        echo "Error: usable Python not found at {{ PYTHON_EXEC }}."; \
        echo "Provision the RTX 5090 / cu128 venv with: just sync"; \
        exit 1; \
    fi

# Full one-time setup: deps + `yopo` on the import path. Run once after cloning.
setup: sync
    @echo "RTX 5090 / cu128 setup complete. Try: just env-doctor"

# Provision the RTX 5090 / cu128 venv from pyproject.toml and put the `yopo`
# package (the repo root) on the venv import path via a .pth, develop-style.
# mmcv is the prebuilt sm_120 wheel; mmengine comes from PyPI.
sync:
    uv sync
    printf '%s\n' "{{ invocation_directory() }}" > "{{ SITE_PACKAGES }}/_yopo_src.pth"
    @echo "RTX 5090 / cu128 deps synced; yopo package on import path (.pth)."

# Read-only environment triage: Python, package imports, GPU.
env-doctor: check-venv
    @echo "=== python ==="
    @"{{ PYTHON_EXEC }}" --version 2>/dev/null || echo "venv missing"
    @echo "=== imports + cuda ==="
    @"{{ PYTHON_EXEC }}" -c "import torch, mmcv, mmengine, yopo; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('torchvision', __import__('torchvision').__version__); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('yopo', yopo.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1 | head -20
    @echo "=== gpu ==="
    @nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

# Generate a tiny synthetic NOCS-format dataset (a handful of frames) so the
# train/infer smokes can run without the full AG-Pose NOCS download.
gen-synthetic: check-venv
    "{{ PYTHON_EXEC }}" scripts/gen_synthetic_nocs.py --out "{{ SMOKE_DATA }}"

# Download the official YOPO R50 NOCS checkpoint into checkpoints/.
download-ckpt:
    mkdir -p checkpoints
    @if [ -f checkpoints/nocs_yopo_real_camera_r50.pth ]; then \
        echo "already present: checkpoints/nocs_yopo_real_camera_r50.pth"; \
    else \
        curl -L -o checkpoints/nocs_yopo_real_camera_r50.pth \
          https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/nocs_yopo_real_camera_r50.pth; \
    fi

# 1-iteration GPU smoke TRAINING on the synthetic NOCS data — proves the real
# YOPO R50 model trains end-to-end (loss -> backward -> checkpoint) on GPU.
smoke-train WORKDIR="work_dirs/smoke_train": check-venv
    "{{ PYTHON_EXEC }}" tools/train.py {{ SMOKE_TRAIN_CONFIG }} --work-dir "{{ WORKDIR }}"

# GPU smoke INFERENCE — builds the real YOPO R50 model, loads a checkpoint (or
# random init), and runs a forward pass on a synthetic frame on GPU.
smoke-infer CKPT="checkpoints/nocs_yopo_real_camera_r50.pth": check-venv
    "{{ PYTHON_EXEC }}" scripts/smoke_infer.py --ckpt "{{ CKPT }}" --data "{{ SMOKE_DATA }}"

# Full evaluation wrapper (needs the real NOCS/HouseCat6D dataset under data/).
test CONFIG CKPT: check-venv
    "{{ PYTHON_EXEC }}" tools/test.py "{{ CONFIG }}" "{{ CKPT }}"

# Full training wrapper (needs the real dataset under data/).
train CONFIG WORKDIR="work_dirs/train": check-venv
    "{{ PYTHON_EXEC }}" tools/train.py "{{ CONFIG }}" --work-dir "{{ WORKDIR }}"

# ============================================================================
# Viewer (opt-in) — Open3D RGB-D point cloud with the predicted 3D ellipsoids,
# and a 2D view of the same ellipsoids projected onto the image.
#
# Open3D is not part of `just sync`: it drags in a GUI/GL stack a training or
# CI box has no use for.  `just viewer-sync` adds it, everything else the
# viewer needs is already a runtime dependency.
#
# The bundle under VIEWER_BUNDLE is generated, not tracked.  Regenerate it for
# whichever checkpoint you want to look at with `just viewer-export`.
# ============================================================================

VIEWER_DIR := "tools/viewer"
VIEWER_BUNDLE := "work_dirs/viewer_bundle"
VIEWER_SCORE := "0.35"

# Install the viewer's extra dependency into the existing venv.
#
# Deliberately `uv pip install` and not `uv sync --group viewer`: a full sync
# re-resolves the whole project and would reinstall mmcv from the sm_120 URL
# this pyproject targets.  A box running a different architecture -- an sm_89
# card with the locally built wheel under wheels/, say -- would lose every
# CUDA op (rotated IoU, rotated NMS, deformable attention) to that swap.  The
# group above stays the declaration of what the viewer needs; this installs
# exactly that and touches nothing else.
#
# On a machine whose GPU matches pyproject, `uv sync --group viewer` is
# equivalent and may be preferred.
viewer-sync: check-venv
    uv pip install --python "{{ PYTHON_EXEC }}" open3d==0.19.0
    @"{{ PYTHON_EXEC }}" -c "import mmcv; from mmcv.ops import box_iou_rotated; \
        import torch; box_iou_rotated( \
            torch.zeros(1, 5, device='cuda'), torch.zeros(1, 5, device='cuda')); \
        print('mmcv CUDA ops still working')" \
        || echo "WARNING: mmcv CUDA ops broke; check the installed wheel"
    @echo "viewer deps installed; run 'just viewer-export CONFIG CKPT' next."

# Fail early with an actionable message rather than an ImportError mid-render.
check-viewer: check-venv
    @"{{ PYTHON_EXEC }}" -c "import open3d" 2>/dev/null \
        || { echo "open3d missing. Run: just viewer-sync"; exit 1; }

# Build a 10-frame bundle (RGB, depth, predictions) from a checkpoint.
# Use the run's own dumped config so anchor settings come along; passing the
# training config from temp/ leaves the depth anchor off.
viewer-export CONFIG CKPT FRAMES="10" NMS="0.20": check-venv
    "{{ PYTHON_EXEC }}" "{{ VIEWER_DIR }}/export_from_yopo.py" \
        "{{ CONFIG }}" "{{ CKPT }}" "{{ VIEWER_BUNDLE }}" \
        --num-frames {{ FRAMES }} --nms-iou-threshold {{ NMS }} \
        --max-predictions 128

# 3D viewer.  Arrow keys move between frames; see tools/viewer/README.md.
viewer SCORE=VIEWER_SCORE: check-viewer
    PYTHONPATH="{{ VIEWER_DIR }}" "{{ PYTHON_EXEC }}" -m gaucho3d_viewer.app \
        --bundle "{{ VIEWER_BUNDLE }}" --score-threshold {{ SCORE }}

# 2D viewer: projected ellipses on the RGB frame, with confidence labels.
viewer-2d SCORE=VIEWER_SCORE: check-venv
    PYTHONPATH="{{ VIEWER_DIR }}" "{{ PYTHON_EXEC }}" -m gaucho3d_viewer.two_d \
        --bundle "{{ VIEWER_BUNDLE }}" --score-threshold {{ SCORE }}

# Static PNGs plus a contact sheet, no GUI required.
viewer-overlays SCORE=VIEWER_SCORE: check-venv
    PYTHONPATH="{{ VIEWER_DIR }}" "{{ PYTHON_EXEC }}" -m gaucho3d_viewer.two_d \
        --bundle "{{ VIEWER_BUNDLE }}" --score-threshold {{ SCORE }} \
        --save-dir "{{ VIEWER_BUNDLE }}/overlays"

# Read-only check that the bundle loads and every frame renders.
viewer-validate SCORE=VIEWER_SCORE: check-viewer
    PYTHONPATH="{{ VIEWER_DIR }}" "{{ PYTHON_EXEC }}" -m gaucho3d_viewer.app \
        --bundle "{{ VIEWER_BUNDLE }}" --score-threshold {{ SCORE }} --validate-only
