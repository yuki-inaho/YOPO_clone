"""Deterministic drawing helpers for projected ellipsoid tracking overlays."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class OpenCVEllipse:
    """Validated OpenCV ellipse arguments derived from ``(a,b,cx,cy,theta)``."""

    center: tuple[int, int]
    axes: tuple[int, int]
    angle_degrees: float


def ellipse_to_cv2(
    ellipse: ArrayLike, *, image_size: tuple[int, int]
) -> OpenCVEllipse | None:
    """Validate a semi-axis-first radian ellipse and convert it for OpenCV."""

    values = np.asarray(ellipse, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError(f"projected ellipse must have shape (5,), got {values.shape}")
    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("image_size must contain positive height and width")
    a, b, center_x, center_y, theta = values.tolist()
    safe_scale = float(max(height, width))
    if (
        not np.isfinite(values).all()
        or min(a, b) <= 0.0
        or max(a, b) > 4.0 * safe_scale
        or abs(center_x) > 8.0 * safe_scale
        or abs(center_y) > 8.0 * safe_scale
    ):
        return None
    return OpenCVEllipse(
        center=(int(np.rint(center_x)), int(np.rint(center_y))),
        axes=(max(1, int(np.rint(a))), max(1, int(np.rint(b)))),
        angle_degrees=float(np.degrees(theta)),
    )


def track_color_bgr(track_id: int) -> tuple[int, int, int]:
    """Return one high-contrast deterministic BGR color for a positive track ID."""

    if track_id <= 0:
        raise ValueError("track_id must be positive")
    hue = (track_id * 137) % 180
    hsv = np.array([[[hue, 210, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def draw_tracked_ellipse(
    image_bgr: np.ndarray,
    ellipse: ArrayLike,
    *,
    track_id: int,
    score: float,
    projected_valid: bool,
) -> bool:
    """Draw one projected ellipse and ID label; return whether it was drawable."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8[H,W,3]")
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and in [0,1]")
    if not projected_valid:
        return False
    converted = ellipse_to_cv2(ellipse, image_size=image_bgr.shape[:2])
    if converted is None:
        return False
    color = track_color_bgr(track_id)
    cv2.ellipse(
        image_bgr,
        converted.center,
        converted.axes,
        converted.angle_degrees,
        0.0,
        360.0,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image_bgr, converted.center, 2, color, -1, cv2.LINE_AA)
    anchor = (converted.center[0] + 3, max(14, converted.center[1] - 4))
    label = f"ID {track_id} {score:.2f}"
    cv2.putText(
        image_bgr,
        label,
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        label,
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )
    return True


__all__ = [
    "OpenCVEllipse",
    "draw_tracked_ellipse",
    "ellipse_to_cv2",
    "track_color_bgr",
]
