"""Evaluate the GauCho-3D ellipsoid under a single, shared correspondence.

Why this metric exists
----------------------
Three different objects were being reported as if they described one system:

* ``EllipseEnvelopeRotatedIoUMetric`` scores ``pred_instances.ellipse_obb`` --
  the *independent 2D* GauCho branch.
* ``NOCSMetric`` scores ``translations / rotations / sizes`` -- the *legacy*
  YOPO cuboid, which predates this work.
* The 3D ellipsoid itself (``ellipsoid_centers`` / ``ellipsoid_shapes``) and its
  perspective projection (``projected_ellipses``) were exposed by the head and
  scored by nothing at all.

Worse, each metric re-matches predictions to ground truth on its own terms, so
the 2D number and the 3D number are not guaranteed to be talking about the same
physical object.  Measured on this data they disagree about a third of the time.

This metric fixes both problems.  It decides the prediction-to-annotation
correspondence **exactly once**, from the projected ellipse, and then computes
every 3D quantity on that fixed pairing.  Nothing here re-matches in 3D, and
nothing here reads the legacy cuboid fields -- ``pred_instances.T``,
``.sizes`` and ``.rotations`` are deliberately never touched, which
``tests/test_gaucho3d_shared_match_metric.py`` pins.

Coordinate frames
-----------------
``projected_ellipses`` is produced by the head from the *original* intrinsic, so
it lives in original-image pixels.  ``gt_instances.obb_gaussians`` lives in the
resized frame the val pipeline produced.  The GT side is mapped forward with
:func:`rescale_compact_gaussian`, the same correction the 2D ellipse metric
already applies; skipping it costs almost all of the recall.

Naming
------
The 3D overlap reported here is between the *envelopes* of the two ellipsoids --
the oriented boxes spanned by their principal axes -- not between the ellipsoids
themselves.  It is named ``envelope_iou`` throughout so it can never be quoted
as a true ellipsoid IoU.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated, nms_rotated
from mmengine.evaluator import BaseMetric
from scipy.optimize import linear_sum_assignment

from yopo.registry import METRICS

from .ellipse_rotated_iou_metric import (compact_gaussian_to_obb,
                                         rescale_compact_gaussian)
from .nocs_metric import compute_ap_from_matches_scores
from .oriented_box_iou_3d import oriented_box_iou_3d
from .rotated_iou_metric import _as_cpu_tensor, _field

__all__ = [
    "GauCho3DSharedMatchMetric",
    "ProjectedEllipsoidRotatedIoUMetric",
    "gt_sigma_from_obb",
    "projected_ellipse_to_rbox",
    "ray_center_errors",
    "sigma_to_envelope_obb",
]

_MM_PER_M = 1000.0
#: Two ellipsoids whose second-best projected IoU is this close to the best are
#: reported separately: the pairing exists but is not trustworthy on its own.
_AMBIGUOUS_MARGIN = 0.05


def projected_ellipse_to_rbox(ellipse: torch.Tensor) -> torch.Tensor:
    """``(a, b, cx, cy, theta)`` to the rotated box ``(cx, cy, w, h, theta)``.

    ``projected_ellipses`` uses the ``gaussian_to_ellipse2d`` layout, which puts
    the semi-axes *first*; ``ellipse_obb`` uses the centre-first layout.  Mixing
    the two silently produces boxes at the wrong place with the wrong size, so
    the conversion is written out here rather than inlined.
    """
    if ellipse.ndim != 2 or ellipse.shape[-1] != 5:
        raise ValueError(
            f"projected ellipses must have shape (N, 5), got "
            f"{tuple(ellipse.shape)}")
    return torch.stack(
        (ellipse[:, 2], ellipse[:, 3], 2.0 * ellipse[:, 0],
         2.0 * ellipse[:, 1], ellipse[:, 4]), dim=-1)


def gt_sigma_from_obb(transform: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Ground-truth ``Sigma`` from the annotated pose and extent.

    ``Sigma = R diag((size/2)^2) R^T`` is the inscribed ellipsoid of the
    annotated oriented box -- the same construction
    :func:`ellipsoid_from_rotation_size` uses to build the training target, so
    the metric scores the object the loss actually optimises.

    ``transform`` is the annotation's 4x4; its linear block is a rigid rotation
    on this data (verified: uniform scale 1.0 to 1e-6), so no scale is stripped.
    """
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    radii_squared = (np.asarray(size, dtype=np.float64) * 0.5) ** 2
    sigma = rotation @ np.diag(radii_squared) @ rotation.T
    return 0.5 * (sigma + sigma.T)


def sigma_to_envelope_obb(center: np.ndarray, sigma: np.ndarray):
    """Return ``(center, size, rotation)`` of the ellipsoid's envelope box.

    The eigenvectors of ``Sigma`` are the ellipsoid's principal axes and the
    square roots of its eigenvalues are the semi-axes, so the smallest oriented
    box containing the ellipsoid has extent ``2 sqrt(lambda)``.  ``eigh``
    returns an orthonormal basis that may be left-handed; one axis is flipped so
    the caller always receives a proper rotation.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    eigenvalues, rotation = np.linalg.eigh(sigma)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    if np.linalg.det(rotation) < 0.0:
        rotation = rotation.copy()
        rotation[:, 0] *= -1.0
    return (np.asarray(center, dtype=np.float64), 2.0 * np.sqrt(eigenvalues),
            rotation)


def ray_center_errors(predicted_center: np.ndarray,
                      target_center: np.ndarray) -> dict:
    """Decompose the centre error along and across the line of sight.

    A plain ``Delta z`` is only the depth error for an object on the optical
    axis.  Off to the side, part of a genuine range error shows up in ``x`` and
    ``y`` instead, and part of a genuine lateral error leaks into ``z``.  The
    camera centre is the origin of this frame, so the correct decomposition is
    along ``r = t_gt / ||t_gt||``.
    """
    predicted_center = np.asarray(predicted_center, dtype=np.float64)
    target_center = np.asarray(target_center, dtype=np.float64)
    target_range = float(np.linalg.norm(target_center))
    if not np.isfinite(target_range) or target_range <= 0.0:
        raise ValueError("target centre must have a positive, finite range")
    ray = target_center / target_range
    error = predicted_center - target_center
    signed = float(ray @ error)
    transverse = error - signed * ray
    predicted_range = float(np.linalg.norm(predicted_center))
    if predicted_range > 0.0:
        cosine = float(np.clip((predicted_center / predicted_range) @ ray,
                               -1.0, 1.0))
        bearing = float(np.degrees(np.arccos(cosine)))
    else:
        bearing = float("nan")
    return {
        "ray_error_signed": signed,
        "ray_error_abs": abs(signed),
        "transverse_error": float(np.linalg.norm(transverse)),
        "bearing_error_deg": bearing,
    }


def _directional_extent(sigma: np.ndarray, direction: np.ndarray) -> float:
    """Full extent of the ellipsoid along a unit direction: ``2 sqrt(d^T S d)``."""
    return 2.0 * float(np.sqrt(max(direction @ sigma @ direction, 0.0)))


def _aspect_ratio(sigma: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)), 1e-18)
    return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))


def _class_aware_rotated_nms(boxes: torch.Tensor, scores: torch.Tensor,
                             labels: torch.Tensor, iou_threshold: float):
    """Rotated NMS that actually separates classes.

    ``mmcv.ops.nms_rotated`` takes a ``labels`` argument but ignores it in this
    version -- two identical boxes with different labels suppress each other.
    Verified, so classes are separated the way ``batched_nms`` does it, by
    translating each class into its own region of the plane.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    boxes = boxes.float()
    scores = scores.float()
    unique_labels = torch.unique(labels)
    if unique_labels.numel() > 1:
        span = boxes[:, :2].abs().max() + boxes[:, 2:4].abs().max()
        shifted = boxes.clone()
        shifted[:, :2] = shifted[:, :2] + labels.to(
            boxes.dtype).unsqueeze(-1) * (span * 2.0 + 1.0)
    else:
        shifted = boxes
    _, keep = nms_rotated(shifted, scores, iou_threshold)
    return keep.reshape(-1).cpu()


@METRICS.register_module()
class ProjectedEllipsoidRotatedIoUMetric(BaseMetric):
    """Rotated AP of the *projected 3D ellipsoid* against the annotated OBB.

    This is the 2D number for the ellipsoid branch, as distinct from
    ``EllipseEnvelopeRotatedIoUMetric``, which scores the independent 2D head.
    Reporting both makes visible whether the 3D branch agrees with the image at
    all before any of its metric-depth claims are believed.
    """

    default_prefix = "projected_ellipsoid"

    def __init__(self,
                 iou_thr: float = 0.5,
                 score_thr: float = 0.05,
                 num_classes: int = 1,
                 nms_iou_threshold: Optional[float] = None,
                 collect_device: str = "cpu",
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = float(iou_thr)
        self.score_thr = float(score_thr)
        self.num_classes = int(num_classes)
        self.nms_iou_threshold = (
            None if nms_iou_threshold is None else float(nms_iou_threshold))

    def process(self, data_batch: dict, data_samples: Sequence[Any]) -> None:
        for data_sample in data_samples:
            prepared = _prepare_frame(data_sample, self.score_thr,
                                      self.nms_iou_threshold)
            self.results.append(
                dict(pred_rboxes=prepared["pred_rboxes"].numpy(),
                     pred_scores=prepared["pred_scores"].numpy(),
                     pred_labels=prepared["pred_labels"].numpy(),
                     gt_rboxes=prepared["gt_rboxes"].numpy(),
                     gt_labels=prepared["gt_labels"].numpy()))

    def compute_metrics(self, results: list) -> dict:
        matches, scores, gt_total = [], [], 0
        for frame in results:
            gt_total += len(frame["gt_labels"])
            frame_match = _greedy_rotated_match(
                torch.from_numpy(frame["pred_rboxes"]),
                torch.from_numpy(frame["pred_scores"]),
                torch.from_numpy(frame["pred_labels"]),
                torch.from_numpy(frame["gt_rboxes"]),
                torch.from_numpy(frame["gt_labels"]), self.iou_thr)
            matches.append(frame_match)
            scores.append(frame["pred_scores"])
        if gt_total == 0:
            return {f"AP_{int(self.iou_thr * 100)}": 0.0}
        pred_match = (np.concatenate(matches) if matches
                      else np.zeros(0, dtype=np.float64))
        pred_scores = (np.concatenate(scores) if scores
                       else np.zeros(0, dtype=np.float64))
        if pred_match.size == 0:
            return {f"AP_{int(self.iou_thr * 100)}": 0.0}
        average_precision = compute_ap_from_matches_scores(
            pred_match, pred_scores, np.zeros(gt_total))
        return {
            f"AP_{int(self.iou_thr * 100)}": float(average_precision),
            "recall": float((pred_match > -1).sum() / gt_total),
        }


def _greedy_rotated_match(pred_rboxes: torch.Tensor, pred_scores: torch.Tensor,
                          pred_labels: torch.Tensor, gt_rboxes: torch.Tensor,
                          gt_labels: torch.Tensor,
                          iou_thr: float) -> np.ndarray:
    """Standard detection matching: score order, one GT per prediction."""
    pred_match = np.full(len(pred_rboxes), -1.0, dtype=np.float64)
    if len(pred_rboxes) == 0 or len(gt_rboxes) == 0:
        return pred_match
    ious = box_iou_rotated(pred_rboxes.float(), gt_rboxes.float()).numpy()
    same_class = (pred_labels.numpy()[:, None] == gt_labels.numpy()[None, :])
    ious = np.where(same_class, ious, -1.0)
    taken = np.zeros(len(gt_rboxes), dtype=bool)
    for index in np.argsort(-pred_scores.numpy(), kind="stable"):
        candidates = np.where(~taken, ious[index], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_thr:
            taken[best] = True
            pred_match[index] = float(best)
    return pred_match


def _prepare_frame(data_sample: Any, score_thr: float,
                   nms_iou_threshold: Optional[float]) -> dict:
    """Score-threshold, NMS and validity-filter one frame, in one place.

    Every downstream field is indexed with the *same* ``keep`` vector, so a
    prediction's projected ellipse, 3D centre, shape and score can never drift
    apart.
    """
    pred = data_sample["pred_instances"]
    gt = data_sample["gt_instances"]

    try:
        projected = _as_cpu_tensor(_field(pred, "projected_ellipses"),
                                   torch.float32)
    except (KeyError, AttributeError) as error:
        raise KeyError(
            "pred_instances.projected_ellipses is missing; enable "
            "gaucho_ellipsoid and expose_gaucho_predictions on the head"
        ) from error
    centers = _as_cpu_tensor(_field(pred, "ellipsoid_centers"), torch.float32)
    sigmas = _as_cpu_tensor(_field(pred, "ellipsoid_shapes"), torch.float32)
    scores = _as_cpu_tensor(_field(pred, "scores"), torch.float32)
    labels = _as_cpu_tensor(_field(pred, "labels"), torch.long)
    try:
        valid = _as_cpu_tensor(_field(pred, "projected_valid"), torch.bool)
    except (KeyError, AttributeError):
        valid = torch.ones(len(projected), dtype=torch.bool)

    finite = (torch.isfinite(projected).all(-1)
              & torch.isfinite(centers).all(-1)
              & torch.isfinite(sigmas).flatten(1).all(-1)
              & (projected[:, :2] > 0).all(-1))
    valid = valid & finite

    keep = scores >= score_thr
    projected, centers, sigmas = projected[keep], centers[keep], sigmas[keep]
    scores, labels, valid = scores[keep], labels[keep], valid[keep]

    rboxes = (projected_ellipse_to_rbox(projected) if len(projected)
              else projected.new_zeros((0, 5)))
    if nms_iou_threshold is not None and len(rboxes):
        # NMS is run on the valid subset only: a degenerate projection has no
        # meaningful box and must not suppress a good neighbour.  The invalid
        # ones are carried through so they still count as false positives.
        order = torch.arange(len(rboxes))
        valid_index = order[valid]
        if len(valid_index):
            survivors = valid_index[_class_aware_rotated_nms(
                rboxes[valid], scores[valid], labels[valid],
                nms_iou_threshold)]
        else:
            survivors = valid_index
        keep_nms = torch.cat((survivors, order[~valid])).sort().values
        projected, centers, sigmas = (projected[keep_nms], centers[keep_nms],
                                      sigmas[keep_nms])
        scores, labels, valid = (scores[keep_nms], labels[keep_nms],
                                 valid[keep_nms])
        rboxes = rboxes[keep_nms]

    gt_compact = _as_cpu_tensor(_field(gt, "obb_gaussians"), torch.float32)
    scale_factor = (data_sample.get("scale_factor")
                    if hasattr(data_sample, "get") else None)
    if scale_factor is not None and gt_compact.numel():
        gt_compact = rescale_compact_gaussian(gt_compact, scale_factor)
    gt_rboxes = (compact_gaussian_to_obb(gt_compact) if gt_compact.numel()
                 else gt_compact.reshape(0, 5))

    return dict(
        pred_rboxes=rboxes,
        pred_scores=scores,
        pred_labels=labels,
        pred_valid=valid,
        pred_centers=centers,
        pred_sigmas=sigmas,
        gt_rboxes=gt_rboxes,
        gt_labels=_as_cpu_tensor(_field(gt, "labels"), torch.long),
        gt_centers=_as_cpu_tensor(_field(gt, "translations"), torch.float32),
        gt_sizes=_as_cpu_tensor(_field(gt, "sizes"), torch.float32),
        gt_transforms=_as_cpu_tensor(_field(gt, "T"), torch.float32),
    )


@METRICS.register_module()
class GauCho3DSharedMatchMetric(BaseMetric):
    """3D ellipsoid quality under one correspondence fixed in the image.

    The pipeline is fixed and never varies with the quantity being reported::

        score >= score_thr
          -> class-aware rotated NMS
          -> IoU(projected ellipse, annotated OBB)
          -> per-class Hungarian
          -> keep pairs with IoU >= match_iou_threshold
          -> every 3D number is computed on those pairs and no others

    Hungarian rather than greedy: the question here is "is the 3D shape right
    for *this* object", which needs the globally consistent assignment, not the
    score-ordered one a detection AP uses.  ``shared_AP_*`` is reported
    alongside so an improvement in the diagnostics cannot hide a collapse in
    detection.
    """

    default_prefix = "gaucho3d"

    def __init__(self,
                 score_thr: float = 0.20,
                 nms_iou_threshold: float = 0.20,
                 match_iou_threshold: float = 0.50,
                 iou_3d_thresholds: Sequence[float] = (0.10, 0.20, 0.25, 0.50),
                 num_classes: int = 1,
                 depth_oracle: bool = False,
                 collect_device: str = "cpu",
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # Replaces only the *range* of each matched prediction with the
        # annotation's, keeping its bearing, its shape and everything
        # else.  The 3D residual is entirely range -- 21.9 mm along the
        # line of sight against 2.2 mm across it, on 18 mm objects -- and
        # the depth channel already holds that answer to within 5.7 mm at
        # the annotated centre.  This says what fixing the depth branch
        # would be worth before the architecture is changed for it.
        self.depth_oracle = bool(depth_oracle)
        if not 0.0 < match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must lie in (0, 1]")
        if nms_iou_threshold is not None and not 0.0 < nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must lie in (0, 1] or be None")
        self.score_thr = float(score_thr)
        self.nms_iou_threshold = (
            None if nms_iou_threshold is None else float(nms_iou_threshold))
        self.match_iou_threshold = float(match_iou_threshold)
        self.iou_3d_thresholds = tuple(float(t) for t in iou_3d_thresholds)
        self.num_classes = int(num_classes)

    def process(self, data_batch: dict, data_samples: Sequence[Any]) -> None:
        for data_sample in data_samples:
            frame = _prepare_frame(data_sample, self.score_thr,
                                   self.nms_iou_threshold)
            self.results.append(
                {key: value.numpy() for key, value in frame.items()})

    # -- correspondence -----------------------------------------------------
    def _match_frame(self, frame: dict):
        """Return the one correspondence table this frame will be scored on."""
        pred_rboxes = torch.from_numpy(frame["pred_rboxes"])
        gt_rboxes = torch.from_numpy(frame["gt_rboxes"])
        valid = frame["pred_valid"]
        num_pred, num_gt = len(pred_rboxes), len(gt_rboxes)
        pairs, margins = [], []
        if num_pred == 0 or num_gt == 0 or not valid.any():
            return pairs, margins, np.zeros((num_pred, num_gt))

        ious = box_iou_rotated(pred_rboxes.float(), gt_rboxes.float()).numpy()
        ious = np.where(valid[:, None], ious, -1.0)
        same_class = (frame["pred_labels"][:, None]
                      == frame["gt_labels"][None, :])
        ious = np.where(same_class, ious, -1.0)

        for label in np.unique(frame["gt_labels"]):
            pred_index = np.where((frame["pred_labels"] == label) & valid)[0]
            gt_index = np.where(frame["gt_labels"] == label)[0]
            if pred_index.size == 0 or gt_index.size == 0:
                continue
            block = ious[np.ix_(pred_index, gt_index)]
            rows, columns = linear_sum_assignment(1.0 - block)
            for row, column in zip(rows, columns):
                iou = float(block[row, column])
                if iou < self.match_iou_threshold:
                    continue
                candidates = np.sort(block[row])[::-1]
                second = float(candidates[1]) if candidates.size > 1 else 0.0
                pairs.append((int(pred_index[row]), int(gt_index[column]), iou))
                margins.append(iou - max(second, 0.0))
        return pairs, margins, ious

    def compute_metrics(self, results: list) -> dict:
        prefix_free: dict = {}
        diagnostics: dict = {key: [] for key in (
            "projected_iou", "ray_error_signed", "ray_error_abs",
            "transverse_error", "bearing_error_deg", "normalized_ray_error",
            "depth_extent_ratio", "pred_aspect", "gt_aspect", "volume_ratio",
            "envelope_iou", "association_margin")}
        pred_match_all, pred_scores_all = [], []
        envelope_iou_all = []
        gt_total = matched_total = invalid_total = pred_total = 0

        for frame in results:
            gt_total += len(frame["gt_labels"])
            pred_total += len(frame["pred_scores"])
            invalid_total += int((~frame["pred_valid"]).sum())
            pairs, margins, _ = self._match_frame(frame)
            matched_total += len(pairs)

            frame_match = np.full(len(frame["pred_scores"]), -1.0)
            frame_envelope = np.zeros(len(frame["pred_scores"]))
            for (pred_index, gt_index, iou), margin in zip(pairs, margins):
                predicted_center = frame["pred_centers"][pred_index].astype(
                    np.float64)
                if self.depth_oracle:
                    target_range = float(np.linalg.norm(
                        frame["gt_centers"][gt_index].astype(np.float64)))
                    predicted_range = float(np.linalg.norm(
                        predicted_center))
                    if predicted_range > 0.0:
                        predicted_center = (predicted_center
                                            / predicted_range
                                            * target_range)
                predicted_sigma = 0.5 * (
                    frame["pred_sigmas"][pred_index].astype(np.float64)
                    + frame["pred_sigmas"][pred_index].astype(np.float64).T)
                target_center = frame["gt_centers"][gt_index].astype(np.float64)
                target_sigma = gt_sigma_from_obb(
                    frame["gt_transforms"][gt_index],
                    frame["gt_sizes"][gt_index])
                if np.linalg.eigvalsh(target_sigma).min() <= 0.0:
                    raise ValueError(
                        "annotated ellipsoid is not positive definite; the "
                        "annotation has a zero or negative extent")

                errors = ray_center_errors(predicted_center, target_center)
                ray = target_center / np.linalg.norm(target_center)
                gt_ray_extent = _directional_extent(target_sigma, ray)
                pred_ray_extent = _directional_extent(predicted_sigma, ray)

                diagnostics["projected_iou"].append(iou)
                diagnostics["association_margin"].append(float(margin))
                diagnostics["ray_error_signed"].append(
                    errors["ray_error_signed"] * _MM_PER_M)
                diagnostics["ray_error_abs"].append(
                    errors["ray_error_abs"] * _MM_PER_M)
                diagnostics["transverse_error"].append(
                    errors["transverse_error"] * _MM_PER_M)
                diagnostics["bearing_error_deg"].append(
                    errors["bearing_error_deg"])
                diagnostics["normalized_ray_error"].append(
                    errors["ray_error_abs"] / max(gt_ray_extent * 0.5, 1e-9))
                diagnostics["depth_extent_ratio"].append(
                    pred_ray_extent / max(gt_ray_extent, 1e-9))
                diagnostics["pred_aspect"].append(_aspect_ratio(predicted_sigma))
                diagnostics["gt_aspect"].append(_aspect_ratio(target_sigma))
                diagnostics["volume_ratio"].append(
                    float(np.sqrt(
                        max(np.linalg.det(predicted_sigma), 1e-30)
                        / max(np.linalg.det(target_sigma), 1e-30))))

                envelope_iou = oriented_box_iou_3d(
                    *sigma_to_envelope_obb(predicted_center, predicted_sigma),
                    *sigma_to_envelope_obb(target_center, target_sigma))
                diagnostics["envelope_iou"].append(float(envelope_iou))
                frame_match[pred_index] = float(gt_index)
                frame_envelope[pred_index] = float(envelope_iou)

            pred_match_all.append(frame_match)
            pred_scores_all.append(frame["pred_scores"])
            envelope_iou_all.append(frame_envelope)

        prefix_free["match_count"] = float(matched_total)
        prefix_free["gt_count"] = float(gt_total)
        prefix_free["prediction_count"] = float(pred_total)
        prefix_free["invalid_prediction_count"] = float(invalid_total)
        prefix_free["match_coverage"] = (
            float(matched_total / gt_total) if gt_total else 0.0)

        for key, values in diagnostics.items():
            if not values:
                prefix_free[f"{key}_median"] = float("nan")
                continue
            array = np.asarray(values, dtype=np.float64)
            prefix_free[f"{key}_median"] = float(np.median(array))
            if key in ("projected_iou", "envelope_iou", "depth_extent_ratio"):
                prefix_free[f"{key}_mean"] = float(np.mean(array))
        if diagnostics["association_margin"]:
            margin = np.asarray(diagnostics["association_margin"])
            prefix_free["ambiguous_match_fraction"] = float(
                np.mean(margin < _AMBIGUOUS_MARGIN))

        if gt_total and pred_match_all:
            pred_match = np.concatenate(pred_match_all)
            pred_scores = np.concatenate(pred_scores_all)
            envelope = np.concatenate(envelope_iou_all)
            for threshold in self.iou_3d_thresholds:
                key = f"{threshold:.2f}".split(".")[1]
                coupled = np.where(
                    (pred_match > -1) & (envelope >= threshold), pred_match,
                    -1.0)
                prefix_free[f"shared_AP_{key}"] = float(
                    compute_ap_from_matches_scores(coupled, pred_scores,
                                                   np.zeros(gt_total)))
                prefix_free[f"envelope_recall_{key}"] = float(
                    (coupled > -1).sum() / gt_total)
        return prefix_free
