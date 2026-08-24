#!/usr/bin/env python3
"""Convert the custom COCO-3D/Omni3D dataset to YOPO NOCS train format.

This takes the WildDet3D-format annotation JSON (single-class ``fruit``,
per-image depth npz in millimetres) and rewrites it into the NOCS ``real``
directory layout that ``NOCSDataset`` (split ``real_train`` / ``real_test``)
loads verbatim::

    data/nocs_custom/real/
      train_list.txt          # lines: <scene>/<frame%04d>
      test_list.txt           # same lines for val split
      <scene>/
        <frame%04d>_color.png     # RGB, 640x480 (uint8)
        <frame%04d>_depth.png     # depth, uint16 *millimetres*
        <frame%04d>_label.pkl     # NOCS train label keys (see below)

Label pkl keys (NOCS train branch):
  class_ids    : np.ndarray (N,) int   1-indexed
  instance_ids : np.ndarray (N,) int
  bboxes       : np.ndarray (N,4) float32  [y1,x1,y2,x2]
  translations : np.ndarray (N,3) float32  metres, camera frame
  rotations    : np.ndarray (N,3,3) float32
  sizes        : np.ndarray (N,3) float32  (width, height, depth)
  scales       : np.ndarray (N,) float32

Mapping from the Omni3D fields:
  center_cam  -> translations
  R_cam       -> rotations  (already right-handed, det>0)
  dimensions  -> sizes      ([w, h, l] as-is)
  bbox (xywh) -> bboxes     (-> [y1,x1,y2,x2])
  category    -> class_ids  (all 1, single class)

Usage::
    pixi run python scripts/gen_yopo_ds.py \
        --data-root /path/to/wilddet3d_custom_data \
        --out data/nocs_custom
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _normalize_pca_rotation(R: list[list[float]]) -> list[list[float]]:
    """Normalize a PCA-derived rotation to a unique Euclidean sign frame.

    PCA eigenvectors have arbitrary column signs. Two otherwise identical
    fruits can therefore come out with mirrored rotation matrices (and mixed
    signs), which makes a 9D pose model learn contradictory labels. We fix a
    canonical sign per column and keep the frame right-handed:

      * column 0: sign chosen so that the largest-magnitude entry is positive.
      * column 1: same rule.
      * column 2: recomputed as ``cross(col0, col1)`` to guarantee det=+1.

    ``dimensions`` ([w, h, l]) is mapped to columns as before, so the extent
    order is preserved.
    """
    M = np.asarray(R, dtype=np.float64)
    col0 = M[:, 0].copy()
    col1 = M[:, 1].copy()

    def _canon_sign(v: np.ndarray) -> np.ndarray:
        j = int(np.argmax(np.abs(v)))
        return v if v[j] >= 0 else -v

    col0 = _canon_sign(col0)
    col1 = _canon_sign(col1)
    col2 = np.cross(col0, col1)  # right-handed frame, det = +1
    norm = np.linalg.norm(col2)
    if norm > 0:
        col2 = col2 / norm
    M = np.stack([col0, col1, col2], axis=1)
    return M.tolist()


# Camera-frame world axes for this sideways rig:
#   WORLD_UP  = cam +x  (image-right) is the zenith (gravity is -x).
#   DEPTH_DIR = cam +z  is the viewing direction (into the scene).
# We anchor the local frame so every fruit's Z (blue) points up (zenith) and
# X (red) points toward the camera viewing (depth) direction; Y is the
# remaining right-handed axis. This makes pose labels self-consistent.
WORLD_UP = np.array([1.0, 0.0, 0.0], dtype=np.float64)
DEPTH_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _align_frame(R: list[list[float]], dims: list[float]) -> tuple[list[list[float]], list[float]]:
    """Anchor the PCA frame to world-up (local Z) and depth (local X).

    ``R`` columns are the PCA principal axes (after sign canonicalization).
    ``dims`` is ``[w, h, l]``. We pick:
      * local Z  = the column most aligned with ``WORLD_UP`` (zenith),
      * local X  = the remaining column most aligned with ``DEPTH_DIR``,
      * local Y  = cross(Z, X) (right-handed).
    All chosen axes are sign-flipped toward their anchor direction. ``dims``
    is re-ordered to follow the new column layout.
    """
    M = np.asarray(R, dtype=np.float64)

    # 1) Z = zenith axis
    up_dots = np.abs(M.T @ WORLD_UP)
    z_idx = int(np.argmax(up_dots))
    z_vec = M[:, z_idx].copy()
    if z_vec @ WORLD_UP < 0:
        z_vec = -z_vec

    # 2) X = depth axis from the remaining two columns
    remaining = [c for c in (0, 1, 2) if c != z_idx]
    depth_dots = {c: abs(float(M[:, c] @ DEPTH_DIR)) for c in remaining}
    x_idx = max(depth_dots, key=depth_dots.get)
    x_vec = M[:, x_idx].copy()
    if x_vec @ DEPTH_DIR < 0:
        x_vec = -x_vec

    # 3) Y = right-handed completion; its physical extent is taken from the
    # leftover original column.
    y_idx = remaining[0] if remaining[0] != x_idx else remaining[1]
    y_vec = np.cross(z_vec, x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)

    # Permute dims: original col -> dim value. dims=[w,h,l] map to
    # original columns as col0=length(dims[2]), col1=height(dims[1]),
    # col2=width(dims[0]).
    col_to_dim = {0: dims[2], 1: dims[1], 2: dims[0]}
    z_dim = col_to_dim[z_idx]
    x_dim = col_to_dim[x_idx]
    y_dim = col_to_dim[y_idx]
    new_dims = [y_dim, x_dim, z_dim]  # [w, h, l]: X shortest->w? keep order

    new_R = np.stack([x_vec, y_vec, z_vec], axis=1)
    return new_R.tolist(), new_dims

SCENE = "scene_1"


def load_labels(ann_path: Path) -> dict:
    """Return image->anns mapping plus the shared image K / size."""
    data = json.loads(ann_path.read_text())
    return data


def write_label_pkl(
    path: Path,
    anns: list[dict],
) -> None:
    """Write a NOCS train label pkl for one frame."""
    n = len(anns)
    if n == 0:
        # Empty frame: still write a valid pkl (NOCS handles N=0 fine).
        empty = np.zeros((0,), dtype=np.float32)
        pkl = {
            "class_ids": np.zeros(0, dtype=np.int32),
            "instance_ids": np.zeros(0, dtype=np.int32),
            "bboxes": empty.reshape(0, 4),
            "translations": empty.reshape(0, 3),
            "rotations": empty.reshape(0, 3, 3),
            "sizes": empty.reshape(0, 3),
            "scales": np.zeros(0, dtype=np.float32),
            "obb_cxcywha_rad": empty.reshape(0, 5),
        }
        with open(path, "wb") as f:
            pickle.dump(pkl, f)
        return

    class_ids = np.ones(n, dtype=np.int32)
    instance_ids = np.arange(1, n + 1, dtype=np.int32)

    bboxes = np.zeros((n, 4), dtype=np.float32)
    translations = np.zeros((n, 3), dtype=np.float32)
    rotations = np.zeros((n, 3, 3), dtype=np.float32)
    sizes = np.zeros((n, 3), dtype=np.float32)
    scales = np.ones(n, dtype=np.float32)
    obbs = np.zeros((n, 5), dtype=np.float32)

    for i, a in enumerate(anns):
        x, y, w, h = a["bbox"]
        bboxes[i] = (y, x, y + h, x + w)
        translations[i] = a["center_cam"]
        R_norm = _normalize_pca_rotation(a["R_cam"])
        R_align, dims_align = _align_frame(
            R_norm, a["dimensions"]
        )
        rotations[i] = R_align
        sizes[i] = dims_align
        obbs[i] = a["obb_cxcywha_rad"]

    pkl = {
        "class_ids": class_ids,
        "instance_ids": instance_ids,
        "bboxes": bboxes,
        "translations": translations,
        "rotations": rotations,
        "sizes": sizes,
        "scales": scales,
        "obb_cxcywha_rad": obbs,
    }
    with open(path, "wb") as f:
        pickle.dump(pkl, f)


def convert_split(
    data_root: Path,
    split: str,
    out_root: Path,
    scene: str,
) -> list[str]:
    """Convert one split (train/val) and return list-file lines.

    Each split writes into its own scene directory (``{scene}_{split}``) so
    frames never collide across splits.
    """
    ann = json.loads(
        (data_root / split / "annotations" / f"custom_{split}.json").read_text()
    )
    annots_by_img: dict[int, list[dict]] = {}
    for a in ann["annotations"]:
        annots_by_img.setdefault(a["image_id"], []).append(a)

    scene_dir = f"{scene}_{split}"
    lines: list[str] = []
    for idx, img in enumerate(sorted(ann["images"], key=lambda x: x["id"])):
        frame = idx  # %04d within the split
        rel_stem = f"{scene_dir}/{frame:04d}"
        img_src = data_root / split / img["file_path"]
        if not img_src.exists():
            continue

        depth_npz = data_root / split / "depth" / split / (
            img["file_path"].split("/")[-1].replace(".jpg", ".npz")
        )
        if not depth_npz.exists():
            continue

        out_dir = out_root / scene_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # RGB
        rgb = Image.open(img_src).convert("RGB")
        rgb.save(out_dir / f"{frame:04d}_color.png")

        # Depth: float32 mm npz -> uint16 mm png
        depth = np.load(depth_npz)["depth"]
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
        Image.fromarray(depth).save(out_dir / f"{frame:04d}_depth.png")

        # Label
        frame_anns = annots_by_img.get(img["id"], [])
        write_label_pkl(out_dir / f"{frame:04d}_label.pkl", frame_anns)

        lines.append(rel_stem)
        if (idx + 1) % 50 == 0:
            print(f"[{split}] {idx + 1}/{len(ann['images'])} frames")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/nocs_custom"))
    parser.add_argument("--scene", default=SCENE)
    args = parser.parse_args()

    out = args.out / "real"
    train_lines = convert_split(args.data_root, "train", out, args.scene)
    val_lines = convert_split(args.data_root, "val", out, args.scene)

    (out / "train_list.txt").write_text("\n".join(train_lines) + "\n")
    (out / "test_list.txt").write_text("\n".join(val_lines) + "\n")

    print(f"train: {len(train_lines)} frames -> {out / 'train_list.txt'}")
    print(f"val  : {len(val_lines)} frames -> {out / 'test_list.txt'}")


if __name__ == "__main__":
    main()
