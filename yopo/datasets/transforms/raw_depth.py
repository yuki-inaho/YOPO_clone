"""Raw-depth transforms for the custom fruit RGB-D data contract."""

from __future__ import annotations

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from yopo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRawDepthImageWithValidMask(BaseTransform):
    """Load uint16 millimetre depth without completion and retain validity."""

    def __init__(self, norm_scale: float = 1000.0) -> None:
        if norm_scale <= 0:
            raise ValueError(f"norm_scale must be positive, got {norm_scale}")
        self.norm_scale = float(norm_scale)

    def transform(self, results: dict) -> dict:
        filename = results["depth_path"]
        raw_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            raise FileNotFoundError(f"failed to load depth image: {filename}")

        if raw_depth.ndim == 2 and raw_depth.dtype == np.uint16:
            depth_mm = raw_depth
        elif raw_depth.ndim == 3:
            # Preserve stock NOCS encoded-depth compatibility without applying
            # any inpainting or numerical substitution.
            depth_mm = raw_depth[:, :, 1].astype(np.uint16) * 256
            depth_mm += raw_depth[:, :, 2].astype(np.uint16)
            depth_mm = np.where(depth_mm == 32001, 0, depth_mm).astype(np.uint16)
        else:
            raise ValueError(
                f"unsupported depth encoding at {filename}: "
                f"shape={raw_depth.shape}, dtype={raw_depth.dtype}"
            )

        results["depth_valid_mask"] = depth_mm > 0
        results["depth"] = depth_mm.astype(np.float32) / self.norm_scale
        return results


@TRANSFORMS.register_module()
class ConcatRawDepthToImage(BaseTransform):
    """Append raw-normalized depth, optionally followed by its validity mask."""

    def __init__(
        self,
        depth_scale: float = 255.0,
        append_valid_mask: bool = False,
        bgr_to_rgb: bool = False,
    ) -> None:
        self.depth_scale = float(depth_scale)
        self.append_valid_mask = append_valid_mask
        self.bgr_to_rgb = bool(bgr_to_rgb)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        depth = results["depth"]
        valid_mask = results["depth_valid_mask"]
        if img.shape[:2] != depth.shape or depth.shape != valid_mask.shape:
            raise ValueError(
                "RGB/depth/mask shape mismatch: "
                f"rgb={img.shape[:2]}, depth={depth.shape}, mask={valid_mask.shape}"
            )

        rgb = img[..., ::-1] if self.bgr_to_rgb else img
        channels = [
            rgb.astype(np.float32, copy=False),
            (depth * self.depth_scale)[..., None],
        ]
        if self.append_valid_mask:
            channels.append(valid_mask.astype(np.float32)[..., None])
        results["img"] = np.concatenate(channels, axis=-1)
        return results
