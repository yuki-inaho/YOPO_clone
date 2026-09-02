"""Interactive RGB-D point-cloud and GauCho-3D ellipsoid viewer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


# OpenCV camera coordinates use +x right, +y down, +z forward.  Open3D's
# conventional interactive view uses +y up and looks toward -z.  The same
# rigid reflection is applied to both the RGB-D points and predicted geometry.
CAMERA_TO_VIEW = np.diag([1.0, -1.0, -1.0])


def _ensure_xdg_runtime_dir() -> None:
    """Provide GLFW with a private runtime dir on minimal desktop sessions."""
    if os.environ.get("XDG_RUNTIME_DIR") or not hasattr(os, "getuid"):
        return
    uid = os.getuid()
    runtime_dir = Path(tempfile.gettempdir()) / f"gaucho3d-open3d-runtime-{uid}"
    if runtime_dir.exists():
        stat = runtime_dir.stat()
        if not runtime_dir.is_dir() or stat.st_uid != uid:
            raise RuntimeError(f"unsafe runtime directory: {runtime_dir}")
    else:
        runtime_dir.mkdir(mode=0o700)
    runtime_dir.chmod(0o700)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)


# GLFW key codes.  ``register_key_callback`` takes the raw GLFW code, and the
# arrow keys have no printable character for ``ord`` to produce, so they have to
# be written out.  Values are from ``glfw3.h`` and are stable across versions.
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_DOWN = 264
_KEY_UP = 265


@dataclass(frozen=True)
class FrameRecord:
    dataset_index: int
    frame_id: str
    color: Path
    depth: Path
    predictions: Path


class Bundle:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        self.depth_scale = float(self.manifest.get("depth_scale", 1000.0))
        self.frames = [
            FrameRecord(
                dataset_index=int(item["dataset_index"]),
                frame_id=str(item["frame_id"]),
                color=self._resolve(item["color"]),
                depth=self._resolve(item["depth"]),
                predictions=self._resolve(item["predictions"]),
            )
            for item in self.manifest["frames"]
        ]
        if not self.frames:
            raise ValueError("bundle contains no frames")
        for frame in self.frames:
            for path in (frame.color, frame.depth, frame.predictions):
                if not path.is_file():
                    raise FileNotFoundError(path)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"intrinsic", "scores", "centers", "sigmas"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in data.files}
    n = len(result["scores"])
    if result["centers"].shape != (n, 3) or result["sigmas"].shape != (n, 3, 3):
        raise ValueError(f"inconsistent ellipsoid arrays in {path}")
    if result["intrinsic"].shape != (3, 3):
        raise ValueError(f"intrinsic must be 3x3 in {path}")
    return result


def _make_point_cloud(
    frame: FrameRecord,
    intrinsic: np.ndarray,
    depth_scale: float,
    depth_trunc: float,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    color = o3d.io.read_image(str(frame.color))
    depth = o3d.io.read_image(str(frame.depth))
    color_array = np.asarray(color)
    depth_array = np.asarray(depth)
    if color_array.ndim != 3 or color_array.shape[2] != 3:
        raise ValueError(f"expected RGB image, got {color_array.shape}: {frame.color}")
    if depth_array.shape != color_array.shape[:2]:
        raise ValueError(
            f"RGB/depth size mismatch {color_array.shape[:2]} vs "
            f"{depth_array.shape}: {frame.frame_id}")
    height, width = depth_array.shape
    camera = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    transform = np.eye(4)
    transform[:3, :3] = CAMERA_TO_VIEW
    cloud.transform(transform)
    return cloud


def _score_color(score: float, threshold: float) -> np.ndarray:
    # Low confidence is orange; high confidence moves through yellow to green.
    t = float(np.clip((score - threshold) / max(1.0 - threshold, 1e-6), 0, 1))
    return np.array([1.0 - 0.75 * t, 0.35 + 0.65 * t, 0.05])


def _ellipsoid_lines(
    centers: np.ndarray,
    sigmas: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    max_ellipsoids: int,
    segments: int = 64,
) -> tuple[o3d.geometry.LineSet, o3d.geometry.PointCloud, int]:
    order = np.argsort(scores)[::-1]
    order = order[scores[order] >= score_threshold][:max_ellipsoids]
    points: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    colors: list[np.ndarray] = []
    displayed_centers: list[np.ndarray] = []
    displayed_colors: list[np.ndarray] = []
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    unit_circles = (
        np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), -1),
        np.stack((np.cos(angles), np.zeros_like(angles), np.sin(angles)), -1),
        np.stack((np.zeros_like(angles), np.cos(angles), np.sin(angles)), -1),
    )

    accepted = 0
    for index in order.tolist():
        center = np.asarray(centers[index], dtype=np.float64)
        sigma = np.asarray(sigmas[index], dtype=np.float64)
        if not np.isfinite(center).all() or not np.isfinite(sigma).all():
            continue
        sigma = (sigma + sigma.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        if eigenvalues.min() <= 0 or eigenvalues.max() > 4.0:
            continue
        radii = np.sqrt(eigenvalues)
        color = _score_color(float(scores[index]), score_threshold)
        transformed_center = CAMERA_TO_VIEW @ center
        displayed_centers.append(transformed_center)
        displayed_colors.append(color)
        for unit_circle in unit_circles:
            ring_camera = center + (unit_circle * radii) @ eigenvectors.T
            ring = ring_camera @ CAMERA_TO_VIEW.T
            offset = len(points)
            points.extend(ring)
            for segment in range(segments):
                lines.append((offset + segment, offset + (segment + 1) % segments))
                colors.append(color)
        accepted += 1

    line_set = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3)),
        o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32).reshape(-1, 2)),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64).reshape(-1, 3))
    center_cloud = o3d.geometry.PointCloud()
    center_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(displayed_centers, dtype=np.float64).reshape(-1, 3))
    center_cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(displayed_colors, dtype=np.float64).reshape(-1, 3))
    return line_set, center_cloud, accepted


class InteractiveViewer:
    def __init__(
        self,
        bundle: Bundle,
        frame_index: int,
        score_threshold: float,
        depth_trunc: float,
        voxel_size: float,
        max_ellipsoids: int,
    ) -> None:
        self.bundle = bundle
        self.frame_index = frame_index % len(bundle.frames)
        self.score_threshold = score_threshold
        self.depth_trunc = depth_trunc
        self.voxel_size = voxel_size
        self.max_ellipsoids = max_ellipsoids
        self.show_cloud = True
        self.show_ellipsoids = True
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def _refresh(self, reset_view: bool = True) -> None:
        frame = self.bundle.frames[self.frame_index]
        prediction = _load_prediction(frame.predictions)
        cloud = _make_point_cloud(
            frame,
            prediction["intrinsic"],
            self.bundle.depth_scale,
            self.depth_trunc,
            self.voxel_size,
        )
        ellipsoids, centers, count = _ellipsoid_lines(
            prediction["centers"],
            prediction["sigmas"],
            prediction["scores"],
            self.score_threshold,
            self.max_ellipsoids,
        )
        self.vis.clear_geometries()
        first = True
        if self.show_cloud:
            self.vis.add_geometry(cloud, reset_bounding_box=reset_view)
            first = False
        if self.show_ellipsoids:
            self.vis.add_geometry(
                ellipsoids, reset_bounding_box=reset_view and first)
            self.vis.add_geometry(centers, reset_bounding_box=False)
            first = False
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(axes, reset_bounding_box=reset_view and first)
        print(
            f"frame {self.frame_index + 1}/{len(self.bundle.frames)} | "
            f"val={frame.dataset_index} | {frame.frame_id} | "
            f"points={len(cloud.points)} | ellipsoids={count} | "
            f"score>={self.score_threshold:.2f}")

    def _next(self, _vis: Any) -> bool:
        self.frame_index = (self.frame_index + 1) % len(self.bundle.frames)
        self._refresh(True)
        return False

    def _previous(self, _vis: Any) -> bool:
        self.frame_index = (self.frame_index - 1) % len(self.bundle.frames)
        self._refresh(True)
        return False

    def _raise_threshold(self, _vis: Any) -> bool:
        self.score_threshold = min(0.95, self.score_threshold + 0.05)
        self._refresh(False)
        return False

    def _lower_threshold(self, _vis: Any) -> bool:
        self.score_threshold = max(0.0, self.score_threshold - 0.05)
        self._refresh(False)
        return False

    def _toggle_cloud(self, _vis: Any) -> bool:
        self.show_cloud = not self.show_cloud
        self._refresh(True)
        return False

    def _toggle_ellipsoids(self, _vis: Any) -> bool:
        self.show_ellipsoids = not self.show_ellipsoids
        self._refresh(True)
        return False

    def _reset(self, _vis: Any) -> bool:
        self._refresh(True)
        return False

    def run(self) -> None:
        if not self.vis.create_window(
                window_name="GauCho-3D RGB-D / Ellipsoid Viewer",
                width=1280, height=800):
            raise RuntimeError("Open3D could not create a window; check DISPLAY")
        self.vis.register_key_callback(ord("N"), self._next)
        self.vis.register_key_callback(ord("P"), self._previous)
        self.vis.register_key_callback(_KEY_RIGHT, self._next)
        self.vis.register_key_callback(_KEY_LEFT, self._previous)
        self.vis.register_key_callback(ord("]"), self._raise_threshold)
        self.vis.register_key_callback(ord("["), self._lower_threshold)
        self.vis.register_key_callback(_KEY_UP, self._raise_threshold)
        self.vis.register_key_callback(_KEY_DOWN, self._lower_threshold)
        self.vis.register_key_callback(ord("D"), self._toggle_cloud)
        self.vis.register_key_callback(ord("E"), self._toggle_ellipsoids)
        self.vis.register_key_callback(ord("R"), self._reset)
        render = self.vis.get_render_option()
        render.background_color = np.array([0.025, 0.025, 0.025])
        render.point_size = 2.0
        render.line_width = 2.0
        self._refresh(True)
        print("keys: \u2192/N next | \u2190/P previous | "
              "\u2191\u2193 or ]/[ score | D point cloud | "
              "E ellipsoids | R reset view | Q/Esc quit")
        self.vis.run()
        self.vis.destroy_window()


def validate_bundle(
    bundle: Bundle,
    depth_trunc: float,
    voxel_size: float,
    score_threshold: float,
    max_ellipsoids: int,
) -> None:
    total_points = 0
    total_ellipsoids = 0
    for frame in bundle.frames:
        prediction = _load_prediction(frame.predictions)
        cloud = _make_point_cloud(
            frame, prediction["intrinsic"], bundle.depth_scale,
            depth_trunc, voxel_size)
        _, _, count = _ellipsoid_lines(
            prediction["centers"], prediction["sigmas"], prediction["scores"],
            score_threshold, max_ellipsoids)
        total_points += len(cloud.points)
        total_ellipsoids += count
        print(f"ok: val={frame.dataset_index:03d} points={len(cloud.points):6d} "
              f"ellipsoids={count:3d} {frame.frame_id}")
    print(f"validated {len(bundle.frames)} frames | points={total_points} | "
          f"ellipsoids={total_ellipsoids}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Repo root is three levels up from tools/viewer/gaucho3d_viewer/.  The
    # bundle is generated output, so it lives under work_dirs/ with everything
    # else that is not tracked -- and emphatically not under data/, which is
    # the dataset.
    default_bundle = (Path(__file__).resolve().parents[3]
                      / "work_dirs" / "viewer_bundle")
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--frame", type=int, default=0,
                        help="initial zero-based bundle frame")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="F1 を最大化する運用点。val 330 枚で "
                             "precision 0.843 / recall 0.698 / F1 0.764")
    parser.add_argument("--depth-trunc", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--max-ellipsoids", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true",
                        help="load every frame and exit without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0, 1]")
    if args.depth_trunc <= 0 or args.voxel_size < 0 or args.max_ellipsoids < 1:
        raise ValueError("depth-trunc/max-ellipsoids must be positive; voxel-size non-negative")
    bundle = Bundle(args.bundle)
    if args.validate_only:
        validate_bundle(
            bundle, args.depth_trunc, args.voxel_size,
            args.score_threshold, args.max_ellipsoids)
        return
    _ensure_xdg_runtime_dir()
    InteractiveViewer(
        bundle,
        args.frame,
        args.score_threshold,
        args.depth_trunc,
        args.voxel_size,
        args.max_ellipsoids,
    ).run()


if __name__ == "__main__":
    main()
