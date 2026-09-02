"""Interactive 2D viewer and static-gallery generator for projected ellipses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app import Bundle, FrameRecord, _load_prediction, _score_color


def _selected_indices(
    prediction: dict[str, np.ndarray], score_threshold: float, max_ellipsoids: int
) -> np.ndarray:
    projected = np.asarray(prediction.get("projected_ellipses"), dtype=np.float64)
    scores = np.asarray(prediction["scores"], dtype=np.float64)
    if projected.ndim != 2 or projected.shape != (len(scores), 5):
        raise ValueError("projected_ellipses must have shape (N, 5)")
    finite = np.isfinite(projected).all(axis=1) & np.isfinite(scores)
    positive = (projected[:, 0] > 0) & (projected[:, 1] > 0)
    order = np.argsort(scores)[::-1]
    return order[finite[order] & positive[order] &
                 (scores[order] >= score_threshold)][:max_ellipsoids]



def _label_font(size: int = 13) -> ImageFont.ImageFont:
    """A small bitmap-safe font for the score labels.

    The default PIL font is fixed at a size too small to read on a 736x512
    frame, so a real TrueType face is preferred when one is installed.  The
    fallback keeps the overlay generator working on a bare system rather than
    failing over a label.
    """
    for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _score_text(score: float) -> str:
    """Two decimals, without the leading zero: the labels sit on 20 px objects."""
    return f"{score:.2f}".lstrip("0") or "0"


def _ellipse_points(ellipse: np.ndarray, samples: int = 144) -> list[tuple[float, float]]:
    """Convert YOPO [semi_a, semi_b, cx, cy, radians] to drawable 2D points."""
    semi_a, semi_b, cx, cy, angle = np.asarray(ellipse, dtype=np.float64)
    phases = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xs = cx + semi_a * np.cos(phases) * cos_a - semi_b * np.sin(phases) * sin_a
    ys = cy + semi_a * np.cos(phases) * sin_a + semi_b * np.sin(phases) * cos_a
    return list(zip(xs.tolist(), ys.tolist()))


def _draw_matplotlib(
    ax: plt.Axes, frame: FrameRecord, prediction: dict[str, np.ndarray],
    score_threshold: float, max_ellipsoids: int, show_scores: bool = True,
) -> int:
    image = np.asarray(Image.open(frame.color).convert("RGB"))
    ax.clear()
    ax.imshow(image)
    indices = _selected_indices(prediction, score_threshold, max_ellipsoids)
    projected = prediction["projected_ellipses"]
    scores = prediction["scores"]
    for index in indices:
        semi_a, semi_b, cx, cy, angle = projected[index]
        color = _score_color(float(scores[index]), score_threshold)
        ax.add_patch(Ellipse(
            (cx, cy), 2.0 * semi_a, 2.0 * semi_b,
            angle=float(np.degrees(angle)), fill=False, linewidth=1.5,
            edgecolor=color))
        ax.plot(cx, cy, marker="+", markersize=4, markeredgewidth=1.0,
                color=color)
        if show_scores:
            # Above the ellipse, outlined: the frames are cluttered and a plain
            # coloured glyph disappears against ripe fruit.
            label = ax.text(
                cx, cy - semi_b - 2.0, _score_text(float(scores[index])),
                color=color, fontsize=6.5, ha="center", va="bottom",
                clip_on=True)
            label.set_path_effects([
                path_effects.Stroke(linewidth=1.6, foreground="black"),
                path_effects.Normal()])
    ax.set_title(
        f"val={frame.dataset_index:03d} | {frame.frame_id} | "
        f"projected ellipsoids={len(indices)} | score ≥ {score_threshold:.2f}")
    ax.set_axis_off()
    return len(indices)


class Interactive2DViewer:
    def __init__(self, bundle: Bundle, frame_index: int, score_threshold: float,
                 max_ellipsoids: int, show_scores: bool = True) -> None:
        self.bundle = bundle
        self.show_scores = show_scores
        self.frame_index = frame_index % len(bundle.frames)
        self.score_threshold = score_threshold
        self.max_ellipsoids = max_ellipsoids
        self.figure, self.axes = plt.subplots(figsize=(12, 8))
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

    def _refresh(self) -> None:
        frame = self.bundle.frames[self.frame_index]
        prediction = _load_prediction(frame.predictions)
        count = _draw_matplotlib(
            self.axes, frame, prediction, self.score_threshold,
            self.max_ellipsoids, self.show_scores)
        self.figure.canvas.draw_idle()
        print(f"frame {self.frame_index + 1}/{len(self.bundle.frames)} | "
              f"val={frame.dataset_index:03d} | ellipses={count} | "
              f"score>={self.score_threshold:.2f} | "
              f"labels={'on' if self.show_scores else 'off'}")

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"n", "right"}:
            self.frame_index = (self.frame_index + 1) % len(self.bundle.frames)
        elif key in {"p", "left"}:
            self.frame_index = (self.frame_index - 1) % len(self.bundle.frames)
        elif key in {"[", "down"}:
            self.score_threshold = max(0.0, self.score_threshold - 0.05)
        elif key in {"]", "up"}:
            self.score_threshold = min(0.95, self.score_threshold + 0.05)
        elif key == "s":
            self.show_scores = not self.show_scores
        else:
            return
        self._refresh()

    def run(self) -> None:
        self._refresh()
        print("keys: N/\u2192 next | P/\u2190 previous | [/] or \u2191\u2193 score | "
              "S score labels | Q/Esc quit")
        plt.show()


def save_overlays(bundle: Bundle, score_threshold: float, max_ellipsoids: int,
                  save_dir: Path, show_scores: bool = True) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    font = _label_font()
    saved: list[Path] = []
    for ordinal, frame in enumerate(bundle.frames):
        prediction = _load_prediction(frame.predictions)
        image = Image.open(frame.color).convert("RGB")
        draw = ImageDraw.Draw(image)
        indices = _selected_indices(prediction, score_threshold, max_ellipsoids)
        for index in indices:
            color = tuple(int(255 * value) for value in _score_color(
                float(prediction["scores"][index]), score_threshold))
            draw.line(_ellipse_points(prediction["projected_ellipses"][index]),
                      fill=color, width=3, joint="curve")
            if show_scores:
                semi_a, semi_b, cx, cy, _ = prediction[
                    "projected_ellipses"][index]
                # Outlined, above the ellipse: a bare coloured glyph is
                # unreadable against ripe fruit.
                draw.text(
                    (float(cx), float(cy) - float(semi_b) - 3.0),
                    _score_text(float(prediction["scores"][index])),
                    fill=color, font=font, anchor="mb",
                    stroke_width=2, stroke_fill=(0, 0, 0))
        out = save_dir / f"{ordinal:02d}_val{frame.dataset_index:03d}_projected_ellipses.png"
        image.save(out)
        saved.append(out)
        print(f"wrote {out.name}: ellipses={len(indices)}")

    # One easy-to-open overview, without hiding the individual full-resolution PNGs.
    thumbs = [Image.open(path).convert("RGB").resize((400, 300)) for path in saved]
    columns = 2
    sheet = Image.new("RGB", (columns * 400, ((len(thumbs) + 1) // columns) * 300), "black")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 400, (index // columns) * 300))
    sheet_path = save_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}")


def validate_bundle(bundle: Bundle, score_threshold: float, max_ellipsoids: int) -> None:
    total = 0
    for frame in bundle.frames:
        prediction = _load_prediction(frame.predictions)
        count = len(_selected_indices(prediction, score_threshold, max_ellipsoids))
        total += count
        print(f"ok: val={frame.dataset_index:03d} projected_ellipses={count:3d} {frame.frame_id}")
    print(f"validated {len(bundle.frames)} frames | projected_ellipses={total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Repo root is three levels up from tools/viewer/gaucho3d_viewer/.  The
    # bundle is generated output, so it lives under work_dirs/ with everything
    # else that is not tracked -- and emphatically not under data/, which is
    # the dataset.
    default_bundle = (Path(__file__).resolve().parents[3]
                      / "work_dirs" / "viewer_bundle")
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="F1 を最大化する運用点。val 330 枚で "
                             "precision 0.843 / recall 0.698 / F1 0.764")
    parser.add_argument("--max-ellipsoids", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--scores", dest="show_scores", action="store_true",
                        default=True,
                        help="各楕円に confidence を描く（既定）")
    labels.add_argument("--no-scores", dest="show_scores",
                        action="store_false",
                        help="confidence の文字を描かない")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0 or args.max_ellipsoids < 1:
        raise ValueError("score-threshold must be in [0, 1]; max-ellipsoids positive")
    bundle = Bundle(args.bundle)
    if args.validate_only:
        validate_bundle(bundle, args.score_threshold, args.max_ellipsoids)
    elif args.save_dir:
        save_overlays(bundle, args.score_threshold, args.max_ellipsoids,
                      args.save_dir, args.show_scores)
    else:
        Interactive2DViewer(
            bundle, args.frame, args.score_threshold, args.max_ellipsoids,
            args.show_scores).run()


if __name__ == "__main__":
    main()
