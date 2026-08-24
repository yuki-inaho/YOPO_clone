#!/usr/bin/env python3
"""Render a demo of the GT pose: original image + 2D OBB + 3D OBB wireframe.

Draws, per instance:
  - 3D OBB wireframe: cuboid from the GT pose (rotation|translation) + size.
    Front edges are drawn thick, back edges thin for depth cueing.
  - coordinate axes (x=red, y=green, z=blue) from the object centre.
  - 2D OBB: the rotated rectangle aligned with the *object's* 3D edge
    directions projected to the image (not the image axes), tightly enclosing
    the 8 projected cuboid corners.
  - a white axis-aligned bbox for reference.

Works directly on the YOPO NOCS dataset layout (data/nocs_custom) so it can be
used without a training run. The output image is a vertically stacked panel:
  [original] [GT pose overlay]
and may be cropped to a region of interest with ``--crop x1,y1,x2,y2``.

Usage::
    .venv/bin/python scripts/demo_pose_vis.py \
        --data-root data/nocs_custom --split real_train --frame 0000 --max-boxes 10 \
        --out /tmp/demo_pose_vis.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

# 640x480 intrinsics used to build the dataset
FX, FY, CX, CY = 443.9066, 449.1953, 321.3503, 230.8687

# 8 corners of a unit box (normalized); scaled by size afterwards.
CUBOID_UNIT = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]
)

# Cuboid face edges: front(0,1,3,2), back(4,5,7,6), struts(0-4,1-5,2-6,3-7)
FRONT_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0)]
BACK_EDGES = [(4, 5), (5, 7), (7, 6), (6, 4)]
STRUT_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

WIRE_COLOR = (85, 221, 85)      # green wireframe
OBB_COLOR = (255, 165, 0)       # orange 2D OBB
AABB_COLOR = (255, 255, 255)    # white reference box
AXIS_COLORS = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}


def _K(intrinsic) -> np.ndarray:
    """Build a 3x3 K from [fx, fy, cx, cy] or a 3x3 array."""
    if len(intrinsic) == 4:
        fx, fy, cx, cy = intrinsic
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return np.asarray(intrinsic, dtype=np.float64)


def project(pts3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N,3) camera points to (N,2) pixels."""
    h = (K @ pts3d.T).T
    return h[:, :2] / h[:, 2:3]


def draw_edges(
    img: np.ndarray,
    pts2d: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw a set of edges between existing projected 2D points."""
    for i, j in edges:
        cv2.line(
            img,
            tuple(pts2d[i].astype(int)),
            tuple(pts2d[j].astype(int)),
            color,
            width,
        )


def render_instances(
    base: np.ndarray,
    Ts: np.ndarray,
    sizes: np.ndarray,
    intrinsic,
    draw_obb: bool = True,
    draw_aabb: bool = True,
    draw_wire: bool = True,
    draw_axis: bool = True,
    draw_id: bool = True,
) -> np.ndarray:
    """Draw all instances onto a copy of ``base`` and return it (RGB)."""
    img = base.copy()
    K = _K(intrinsic)

    for tidx in range(len(Ts)):
        R = Ts[tidx][:3, :3]
        t = Ts[tidx][:3, 3]
        width, height, depth = sizes[tidx]

        corners_local = CUBOID_UNIT * np.array([width, height, depth])
        corners_cam = (R @ corners_local.T).T + t
        corners2d = project(corners_cam, K)
        center2d = project(t[None], K)[0]

        # (a) 3D wireframe with implicit front/back depth cueing.
        # Front edges thick, back/struts thin -> reads as a solid box.
        if draw_wire:
            draw_edges(img, corners2d, FRONT_EDGES, WIRE_COLOR, 3)
            draw_edges(img, corners2d, STRUT_EDGES, WIRE_COLOR, 2)
            draw_edges(img, corners2d, BACK_EDGES, WIRE_COLOR, 1)

        # (b) coordinate axes (x=red, y=green, z=blue) - thin so they don't
        # fight the box outline.
        if draw_axis:
            axis_len = 0.5 * min(width, height, depth) + 0.01
            for name, col in (("x", 0), ("y", 1), ("z", 2)):
                tip3d = t + axis_len * R[:, col]
                tip2d = project(tip3d[None], K)[0]
                cv2.line(
                    img,
                    tuple(center2d.astype(int)),
                    tuple(tip2d.astype(int)),
                    AXIS_COLORS[name],
                    1,
                )

        # (c) 2D OBB: rotated rect enclosing the projected cuboid corners
        # (minAreaRect -> aligned with the dominant projected object axis).
        if draw_obb:
            rect = cv2.minAreaRect(corners2d.astype(np.float32))
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(img, [box], True, OBB_COLOR, 2)

        # (d) reference axis-aligned bbox
        if draw_aabb:
            x1, y1 = corners2d[:, 0].min(), corners2d[:, 1].min()
            x2, y2 = corners2d[:, 0].max(), corners2d[:, 1].max()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), AABB_COLOR, 1)

        # (e) instance id label
        if draw_id:
            x1, y1 = corners2d[:, 0].min(), corners2d[:, 1].min()
            cv2.putText(
                img,
                f"#{tidx}",
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


def load_frame(
    data_root: Path, split: str, frame: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb, translations, rotations, sizes) for one frame."""
    real = data_root / "real"
    scene = f"scene_1_{'train' if 'train' in split else 'val'}"
    color = real / scene / f"{frame}_color.png"
    label = real / scene / f"{frame}_label.pkl"

    rgb = cv2.imread(str(color))[:, :, ::-1]
    if rgb is None:
        raise SystemExit(f"cannot read {color}")
    with open(label, "rb") as f:
        pkl = pickle.load(f)
    return rgb, pkl["translations"], pkl["rotations"], pkl["sizes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nocs_custom"))
    parser.add_argument("--split", default="real_train")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--out", type=Path, default=Path("/tmp/demo_pose_vis.png"))
    parser.add_argument("--crop", type=str, default=None,
                        help="x1,y1,x2,y2 to zoom into a region")
    parser.add_argument("--max-boxes", type=int, default=0,
                        help="limit drawn instances (0 = all)")
    parser.add_argument("--no-obb", action="store_true")
    parser.add_argument("--no-aabb", action="store_true")
    parser.add_argument("--no-wire", action="store_true")
    parser.add_argument("--no-axis", action="store_true")
    args = parser.parse_args()

    rgb, translations, rotations, sizes = load_frame(
        args.data_root, args.split, args.frame
    )
    intrinsic = [FX, FY, CX, CY]

    n = len(translations)
    n_show = args.max_boxes if args.max_boxes > 0 else n
    if n_show < n:
        print(f"showing {n_show}/{n} instances")

    Ts = np.concatenate(
        [rotations.reshape(n, 3, 3), translations.reshape(n, 3, 1)], axis=2
    )

    # Panel 1 (top): 2D OBB only (+ reference AABB) - shows the 2D rotated
    # boxes on the original image.
    panel_obb = render_instances(
        rgb,
        Ts[:n_show],
        sizes[:n_show],
        intrinsic,
        draw_obb=not args.no_obb,
        draw_aabb=not args.no_aabb,
        draw_wire=False,
        draw_axis=False,
        draw_id=False,
    )
    # Panel 2 (bottom): 3D BBOX wireframe + thin coordinate axes.
    panel_3d = render_instances(
        rgb,
        Ts[:n_show],
        sizes[:n_show],
        intrinsic,
        draw_obb=False,
        draw_aabb=False,
        draw_wire=not args.no_wire,
        draw_axis=not args.no_axis,
        draw_id=False,
    )

    label = np.zeros((26, rgb.shape[1], 3), dtype=np.uint8)
    label[:] = (30, 30, 30)
    text = f"{args.split} {args.frame}: top=2D OBB | bottom=3D BBOX (#inst {n_show})"
    cv2.putText(label, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    canvas = np.vstack([label, panel_obb, panel_3d])

    if args.crop:
        x1, y1, x2, y2 = map(int, args.crop.split(","))
        c_obb = panel_obb[y1:y2, x1:x2]
        c_3d = panel_3d[y1:y2, x1:x2]
        c_label = np.zeros((26, x2 - x1, 3), dtype=np.uint8)
        c_label[:] = (30, 30, 30)
        text = f"{args.split} {args.frame} crop[{x1},{y1}->{x2},{y2}] (top=2D OBB, bottom=3D BBOX)"
        cv2.putText(c_label, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        canvas = np.vstack([c_label, c_obb, c_3d])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas[:, :, ::-1])
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
