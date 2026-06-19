"""Generate a minimal synthetic NOCS dataset for smoke-testing.

Produces a directory tree that NOCSDataset can load verbatim for both
training splits (camera_train / real_train) and the test split (real_test).
Only numpy and Pillow/opencv are used -- no torch, mmcv, or mmengine.

Directory layout produced under --out (default data/nocs_smoke):

  data/nocs_smoke/
    real/
      train_list.txt          # lines like: scene_1/0000
      test_list.txt           # same four frames (re-used for simplicity)
      scene_1/
        0000_color.png        # H=480 W=640 uint8 RGB
        0000_depth.png        # uint16, millimetres
        0000_label.pkl        # train label (keys below)
        0001_color.png
        ...
    camera/
      train_list.txt
      val_list.txt
      scene_1/
        0000_color.png
        0000_label.pkl        # same train label format
        ...
    camera_full_depths/
      scene_1/
        0000_composed.png     # uint16 depth for camera split
        ...
    segmentation_results/
      REAL275/
        results_test_scene_1_0000.pkl  # test label (keys below)
        ...
      CAMERA25/
        results_val_scene_1_0000.pkl   # val label (keys below)
        ...

Label PKL keys (train):
  class_ids     : np.ndarray (N,) int, 1-indexed (1..6)
  instance_ids  : np.ndarray (N,) int
  bboxes        : np.ndarray (N,4) float32, [y1,x1,y2,x2]
  translations  : np.ndarray (N,3) float32, metres in camera frame
  rotations     : np.ndarray (N,3,3) float32, rotation matrices
  sizes         : np.ndarray (N,3) float32, normalised NOCS size (metres-ish)
  scales        : np.ndarray (N,) float32, per-instance scale factors

Label PKL keys (test / segmentation_results):
  gt_class_ids        : np.ndarray (N,) int, 1-indexed
  gt_bboxes           : np.ndarray (N,4) float32, [y1,x1,y2,x2]
  gt_RTs              : np.ndarray (N,4,4) float32
  gt_scales           : np.ndarray (N,3) float32
  gt_handle_visibility: np.ndarray (N,) float32

Intrinsics stored in SPLIT_INFO (not in pkl):
  real split : [591.0125, 590.16775, 322.525, 244.11084]  (fx,fy,cx,cy)
  camera split: [577.5, 577.5, 319.5, 239.5]

Category id mapping (1-indexed, as stored in label pkl):
  1=bottle  2=bowl  3=camera  4=can  5=laptop  6=mug
  (0-indexed inside the model: subtract 1 → 0..5)
"""

import argparse
import os
import pickle
import sys

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

if not _PIL_AVAILABLE and not _CV2_AVAILABLE:
    sys.exit("Either Pillow or opencv-python must be installed.")

# ── Constants ────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 480, 640
CATEGORIES = ("bottle", "bowl", "camera", "can", "laptop", "mug")
# 1-indexed class ids that appear in the pkl files
CAT_ID_MAP = {name: i + 1 for i, name in enumerate(CATEGORIES)}
RNG_SEED = 42

# Symmetric class ids (0-indexed) used internally — kept for documentation
SYM_IDS_0IDX = [0, 1, 3]  # bottle, bowl, can

# Intrinsics [fx, fy, cx, cy]
INTRINSIC_REAL = [591.0125, 590.16775, 322.525, 244.11084]
INTRINSIC_CAMERA = [577.5, 577.5, 319.5, 239.5]


# ── Image writers ─────────────────────────────────────────────────────────────

def _write_rgb(path: str, arr: np.ndarray) -> None:
    """Write H×W×3 uint8 array as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _PIL_AVAILABLE:
        Image.fromarray(arr.astype(np.uint8), "RGB").save(path)
    else:
        cv2.imwrite(path, cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _write_depth_uint16(path: str, arr: np.ndarray) -> None:
    """Write H×W uint16 depth image as PNG (millimetres)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _PIL_AVAILABLE:
        Image.fromarray(arr.astype(np.uint16), "I;16").save(path)
    else:
        cv2.imwrite(path, arr.astype(np.uint16))


# ── Data generators ──────────────────────────────────────────────────────────

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3×3 rotation matrix via QR decomposition."""
    A = rng.standard_normal((3, 3)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Make det(Q) = +1
    Q = Q * np.sign(np.linalg.det(Q))
    return Q.astype(np.float32)


def _make_train_label(rng: np.random.Generator, n_inst: int = 2) -> dict:
    """Create a synthetic training label pickle dict.

    Keys required by NOCSDataset._parse_instance_info (train branch):
      class_ids, instance_ids, bboxes, translations, rotations, sizes, scales
    """
    # Class ids: 1-indexed, pick n_inst from the 6 categories
    class_ids = rng.integers(1, 7, size=n_inst).astype(np.int32)

    # Bboxes: [y1, x1, y2, x2] format (as parsed by dataset)
    # Keep boxes safely within image bounds
    y1s = rng.integers(10, 200, size=n_inst).astype(np.float32)
    x1s = rng.integers(10, 300, size=n_inst).astype(np.float32)
    ys_hw = rng.integers(50, 150, size=n_inst).astype(np.float32)
    xs_hw = rng.integers(50, 150, size=n_inst).astype(np.float32)
    bboxes = np.stack([y1s, x1s, y1s + ys_hw, x1s + xs_hw], axis=1)

    # 3D translations: z in [0.5, 2.0] m
    z = rng.uniform(0.5, 2.0, size=n_inst).astype(np.float32)
    x = rng.uniform(-0.3, 0.3, size=n_inst).astype(np.float32)
    y = rng.uniform(-0.2, 0.2, size=n_inst).astype(np.float32)
    translations = np.stack([x, y, z], axis=1).astype(np.float32)

    # Rotation matrices
    rotations = np.stack([_random_rotation(rng) for _ in range(n_inst)])

    # Normalised NOCS sizes (rough typical values in metres)
    sizes = rng.uniform(0.05, 0.30, size=(n_inst, 3)).astype(np.float32)

    # Per-instance scale factors (usually ~1.0)
    scales = rng.uniform(0.8, 1.2, size=n_inst).astype(np.float32)

    instance_ids = np.arange(1, n_inst + 1, dtype=np.int32)

    return dict(
        class_ids=class_ids,
        instance_ids=instance_ids,
        bboxes=bboxes,
        translations=translations,
        rotations=rotations,
        sizes=sizes,
        scales=scales,
    )


def _make_test_label(rng: np.random.Generator, n_inst: int = 2) -> dict:
    """Create a synthetic test label pickle dict (segmentation_results format).

    Keys required by NOCSDataset._parse_instance_info (test branch):
      gt_class_ids, gt_bboxes, gt_RTs, gt_scales, gt_handle_visibility
    """
    gt_class_ids = rng.integers(1, 7, size=n_inst).astype(np.int32)

    y1s = rng.integers(10, 200, size=n_inst).astype(np.float32)
    x1s = rng.integers(10, 300, size=n_inst).astype(np.float32)
    ys_hw = rng.integers(50, 150, size=n_inst).astype(np.float32)
    xs_hw = rng.integers(50, 150, size=n_inst).astype(np.float32)
    gt_bboxes = np.stack([y1s, x1s, y1s + ys_hw, x1s + xs_hw], axis=1)

    # gt_RTs: 4×4 homogeneous transforms
    gt_RTs = np.zeros((n_inst, 4, 4), dtype=np.float32)
    for i in range(n_inst):
        R = _random_rotation(rng)
        t = np.array([
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.2, 0.2),
            rng.uniform(0.5, 2.0),
        ], dtype=np.float32)
        gt_RTs[i, :3, :3] = R
        gt_RTs[i, :3, 3] = t
        gt_RTs[i, 3, 3] = 1.0

    gt_scales = rng.uniform(0.05, 0.30, size=(n_inst, 3)).astype(np.float32)

    # Handle visibility: 1 = visible, 0 = occluded (relevant for mug)
    gt_handle_visibility = np.ones(n_inst, dtype=np.float32)

    return dict(
        gt_class_ids=gt_class_ids,
        gt_bboxes=gt_bboxes,
        gt_RTs=gt_RTs,
        gt_scales=gt_scales,
        gt_handle_visibility=gt_handle_visibility,
    )


def _frame_stem(frame_idx: int) -> str:
    """Zero-padded 4-digit frame stem, e.g. '0003'."""
    return f"{frame_idx:04d}"


# ── Main generator ────────────────────────────────────────────────────────────

def generate(out_dir: str, n_frames: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(out_dir, exist_ok=True)

    scene = "scene_1"

    # ── Shared synthetic colour & depth images ────────────────────────────────
    # We generate once and reuse across splits; smoke tests only need valid files.
    rgb_imgs = []
    depth_imgs = []
    for fi in range(n_frames):
        rgb = rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
        # depth in mm, roughly 0.5–2.5 m range
        depth = rng.integers(500, 2500, (IMG_H, IMG_W), dtype=np.uint16)
        rgb_imgs.append(rgb)
        depth_imgs.append(depth)

    # ── real/ split ──────────────────────────────────────────────────────────
    real_train_dir = os.path.join(out_dir, "real", scene)
    os.makedirs(real_train_dir, exist_ok=True)

    train_lines = []
    test_lines = []
    for fi in range(n_frames):
        stem = _frame_stem(fi)
        frame_key = f"{scene}/{stem}"

        color_path = os.path.join(real_train_dir, f"{stem}_color.png")
        depth_path = os.path.join(real_train_dir, f"{stem}_depth.png")
        label_path = os.path.join(real_train_dir, f"{stem}_label.pkl")

        _write_rgb(color_path, rgb_imgs[fi])
        _write_depth_uint16(depth_path, depth_imgs[fi])

        train_lbl = _make_train_label(rng)
        with open(label_path, "wb") as f:
            pickle.dump(train_lbl, f)

        train_lines.append(frame_key)
        test_lines.append(frame_key)

    # train_list.txt
    with open(os.path.join(out_dir, "real", "train_list.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")

    # test_list.txt (same frames reused as test)
    with open(os.path.join(out_dir, "real", "test_list.txt"), "w") as f:
        f.write("\n".join(test_lines) + "\n")

    # segmentation_results/REAL275 -- one pkl per frame for real_test
    seg_real_dir = os.path.join(out_dir, "segmentation_results", "REAL275")
    os.makedirs(seg_real_dir, exist_ok=True)
    for fi in range(n_frames):
        stem = _frame_stem(fi)
        seg_path = os.path.join(seg_real_dir, f"results_test_{scene}_{stem}.pkl")
        test_lbl = _make_test_label(rng)
        with open(seg_path, "wb") as f:
            pickle.dump(test_lbl, f)

    # ── camera/ split ────────────────────────────────────────────────────────
    cam_train_dir = os.path.join(out_dir, "camera", scene)
    os.makedirs(cam_train_dir, exist_ok=True)

    cam_train_lines = []
    cam_val_lines = []
    for fi in range(n_frames):
        stem = _frame_stem(fi)
        frame_key = f"{scene}/{stem}"

        color_path = os.path.join(cam_train_dir, f"{stem}_color.png")
        label_path = os.path.join(cam_train_dir, f"{stem}_label.pkl")

        _write_rgb(color_path, rgb_imgs[fi])

        train_lbl = _make_train_label(rng)
        with open(label_path, "wb") as f:
            pickle.dump(train_lbl, f)

        cam_train_lines.append(frame_key)
        cam_val_lines.append(frame_key)

    with open(os.path.join(out_dir, "camera", "train_list.txt"), "w") as f:
        f.write("\n".join(cam_train_lines) + "\n")
    with open(os.path.join(out_dir, "camera", "val_list.txt"), "w") as f:
        f.write("\n".join(cam_val_lines) + "\n")

    # camera_full_depths/ -- composed uint16 depth (camera split depth source)
    cam_depth_dir = os.path.join(out_dir, "camera_full_depths", scene)
    os.makedirs(cam_depth_dir, exist_ok=True)
    for fi in range(n_frames):
        stem = _frame_stem(fi)
        composed_path = os.path.join(cam_depth_dir, f"{stem}_composed.png")
        _write_depth_uint16(composed_path, depth_imgs[fi])

    # segmentation_results/CAMERA25 -- val labels for camera_val
    seg_cam_dir = os.path.join(out_dir, "segmentation_results", "CAMERA25")
    os.makedirs(seg_cam_dir, exist_ok=True)
    for fi in range(n_frames):
        stem = _frame_stem(fi)
        seg_path = os.path.join(seg_cam_dir, f"results_val_{scene}_{stem}.pkl")
        val_lbl = _make_test_label(rng)
        with open(seg_path, "wb") as f:
            pickle.dump(val_lbl, f)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"Generated {n_frames} synthetic frames under: {os.path.abspath(out_dir)}")
    print("  real/scene_1/          : color + depth + train label pkls")
    print("  camera/scene_1/        : color + train label pkls")
    print("  camera_full_depths/    : composed depth for camera split")
    print("  segmentation_results/  : test/val label pkls")
    print(f"  real/train_list.txt    : {n_frames} entries")
    print(f"  real/test_list.txt     : {n_frames} entries")
    print(f"  camera/train_list.txt  : {n_frames} entries")
    print(f"  camera/val_list.txt    : {n_frames} entries")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        default="data/nocs_smoke",
        help="Root output directory for synthetic data (default: data/nocs_smoke)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of synthetic frames to generate per scene/split (default: 4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(out_dir=args.out, n_frames=args.n)
