"""Tests for the shared-correspondence GauCho-3D metric.

These pin the two properties the metric exists to guarantee: the
correspondence is decided once in the image and never re-decided in 3D, and the
legacy YOPO cuboid fields are never consulted.  Both were violated by the
metrics this one replaces.
"""

import numpy as np
import pytest
import torch

from yopo.evaluation.metrics.gaucho3d_shared_match_metric import (
    GauCho3DSharedMatchMetric, gt_sigma_from_obb, projected_ellipse_to_rbox,
    ray_center_errors, sigma_to_envelope_obb)
from yopo.models.losses.gaucho3d_loss import Ellipsoid3DKLDLoss


class _Tripwire(dict):
    """A ``pred_instances`` stand-in that records forbidden field reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touched = []

    def __getitem__(self, key):
        if key in ("T", "sizes", "rotations"):
            self.touched.append(key)
        return super().__getitem__(key)


def _sigma(size, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return rotation @ np.diag((np.asarray(size) * 0.5) ** 2) @ rotation.T


def _sample(centers, sigmas, gt_centers, gt_sizes, *, scores=None,
            projected=None, valid=None, gt_rotations=None):
    """Build one evaluator-form data sample.

    When ``projected`` is not given the ellipses are placed so that prediction
    ``i`` overlaps ground truth ``i`` exactly, which makes a mismatch in the
    correspondence table unambiguous evidence of a bug rather than of noise.
    """
    count = len(centers)
    if projected is None:
        # (a, b, cx, cy, theta): a 10 px circle at x = 100 * i.
        projected = torch.tensor(
            [[10.0, 10.0, 100.0 * (i + 1), 50.0, 0.0] for i in range(count)])
    if scores is None:
        scores = torch.full((count,), 0.9)
    if valid is None:
        valid = torch.ones(count, dtype=torch.bool)

    num_gt = len(gt_centers)
    # The annotated OBB as a compact Gaussian, co-located with prediction i.
    gt_compact = torch.tensor(
        [[100.0 * (i + 1), 50.0, 100.0, 0.0, 100.0] for i in range(num_gt)])

    transforms = torch.eye(4).repeat(num_gt, 1, 1)
    if gt_rotations is not None:
        for i, rotation in enumerate(gt_rotations):
            transforms[i, :3, :3] = torch.as_tensor(rotation,
                                                    dtype=torch.float32)
    for i, center in enumerate(gt_centers):
        transforms[i, :3, 3] = torch.as_tensor(center, dtype=torch.float32)

    pred = _Tripwire(
        projected_ellipses=projected,
        projected_valid=valid,
        ellipsoid_centers=torch.as_tensor(centers, dtype=torch.float32),
        ellipsoid_shapes=torch.as_tensor(np.asarray(sigmas),
                                         dtype=torch.float32),
        scores=scores,
        labels=torch.zeros(count, dtype=torch.long),
        # Legacy fields: present, so that "not read" is a real property and not
        # an accident of them being absent.
        T=torch.eye(4).repeat(count, 1, 1),
        sizes=torch.zeros(count, 3),
        rotations=torch.zeros(count, 6),
    )
    gt = dict(
        obb_gaussians=gt_compact,
        labels=torch.zeros(num_gt, dtype=torch.long),
        translations=torch.as_tensor(gt_centers, dtype=torch.float32),
        sizes=torch.as_tensor(gt_sizes, dtype=torch.float32),
        T=transforms,
    )
    return {"pred_instances": pred, "gt_instances": gt}, pred


def _run(samples, **kwargs):
    metric = GauCho3DSharedMatchMetric(**kwargs)
    metric.process({}, [s for s, _ in samples])
    return metric.compute_metrics(metric.results)


# -- 1. exact agreement --------------------------------------------------


def test_perfect_prediction_scores_perfectly():
    size = [0.02, 0.02, 0.02]
    sample, _ = _sample(centers=[[0.0, 0.0, 0.4]], sigmas=[_sigma(size)],
                        gt_centers=[[0.0, 0.0, 0.4]], gt_sizes=[size])
    out = _run([(sample, None)])
    assert out["match_count"] == 1.0
    assert out["match_coverage"] == pytest.approx(1.0)
    assert out["envelope_iou_median"] == pytest.approx(1.0, abs=1e-6)
    assert out["ray_error_abs_median"] == pytest.approx(0.0, abs=1e-6)
    assert out["transverse_error_median"] == pytest.approx(0.0, abs=1e-6)
    assert out["depth_extent_ratio_median"] == pytest.approx(1.0, abs=1e-6)
    assert out["shared_AP_20"] == pytest.approx(1.0, abs=1e-6)
    assert out["envelope_recall_50"] == pytest.approx(1.0)


# -- 2. the 3D side must not re-decide the correspondence ----------------


def test_swapped_3d_centres_keep_the_image_correspondence():
    """Prediction 0 sits at GT 1's depth and vice versa.

    A metric that re-matched in 3D would pair them across and report a small
    error.  Holding the image correspondence must instead expose the swap.
    """
    size = [0.02, 0.02, 0.02]
    sample, _ = _sample(
        centers=[[0.0, 0.0, 0.8], [0.0, 0.0, 0.4]],
        sigmas=[_sigma(size), _sigma(size)],
        gt_centers=[[0.0, 0.0, 0.4], [0.0, 0.0, 0.8]],
        gt_sizes=[size, size])
    out = _run([(sample, None)])
    assert out["match_count"] == 2.0
    # 0.4 m of range error each way, not the ~0 a 3D re-match would report.
    assert out["ray_error_abs_median"] == pytest.approx(400.0, abs=1e-3)
    assert out["envelope_iou_median"] == pytest.approx(0.0, abs=1e-9)


# -- 3. the legacy cuboid fields are never consulted ---------------------


def test_legacy_cuboid_fields_are_never_read():
    size = [0.02, 0.02, 0.02]
    sample, pred = _sample(centers=[[0.0, 0.0, 0.4]], sigmas=[_sigma(size)],
                           gt_centers=[[0.0, 0.0, 0.4]], gt_sizes=[size])
    _run([(sample, None)])
    assert pred.touched == [], (
        f"metric read legacy prediction fields {pred.touched}; it must score "
        "the ellipsoid, not the YOPO cuboid")


# -- 4. line-of-sight decomposition off the optical axis -----------------


def test_ray_decomposition_is_relative_to_the_line_of_sight():
    """An object at 45 degrees: a pure +z error is not a pure range error."""
    target = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    errors = ray_center_errors(target + np.array([0.0, 0.0, 1.0]), target)
    # The +z step splits evenly along and across a 45-degree line of sight.
    assert errors["ray_error_signed"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert errors["transverse_error"] == pytest.approx(1.0 / np.sqrt(2.0))
    # Pure range error stays entirely in the ray component.
    along = ray_center_errors(target * 1.5, target)
    assert along["transverse_error"] == pytest.approx(0.0, abs=1e-12)
    assert along["bearing_error_deg"] == pytest.approx(0.0, abs=1e-9)


# -- 5. invalid projections are counted, not silently dropped ------------


def test_invalid_projection_is_counted_and_never_matched():
    size = [0.02, 0.02, 0.02]
    sample, _ = _sample(
        centers=[[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]],
        sigmas=[_sigma(size), _sigma(size)],
        gt_centers=[[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]],
        gt_sizes=[size, size],
        valid=torch.tensor([True, False]))
    out = _run([(sample, None)])
    assert out["invalid_prediction_count"] == 1.0
    assert out["match_count"] == 1.0
    # It still occupies a detection slot, so recall cannot reach 1.
    assert out["envelope_recall_50"] == pytest.approx(0.5)


def test_non_finite_projection_is_treated_as_invalid():
    size = [0.02, 0.02, 0.02]
    projected = torch.tensor([[10.0, 10.0, 100.0, 50.0, 0.0],
                              [float("nan"), 10.0, 200.0, 50.0, 0.0]])
    sample, _ = _sample(
        centers=[[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]],
        sigmas=[_sigma(size), _sigma(size)],
        gt_centers=[[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]],
        gt_sizes=[size, size], projected=projected)
    assert _run([(sample, None)])["invalid_prediction_count"] == 1.0


# -- 6. a degenerate annotation is an error, not a silent zero -----------


def test_degenerate_annotation_fails_loudly():
    size = [0.02, 0.02, 0.02]
    sample, _ = _sample(centers=[[0.0, 0.0, 0.4]], sigmas=[_sigma(size)],
                        gt_centers=[[0.0, 0.0, 0.4]],
                        gt_sizes=[[0.02, 0.02, 0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        _run([(sample, None)])


# -- 7. the acceptance threshold is exactly where it claims to be --------


def test_match_threshold_boundary():
    size = [0.02, 0.02, 0.02]

    def coverage(shift):
        projected = torch.tensor([[10.0, 10.0, 100.0 + shift, 50.0, 0.0]])
        sample, _ = _sample(centers=[[0.0, 0.0, 0.4]], sigmas=[_sigma(size)],
                            gt_centers=[[0.0, 0.0, 0.4]], gt_sizes=[size],
                            projected=projected)
        return _run([(sample, None)])["match_count"]

    assert coverage(0.0) == 1.0      # identical boxes, IoU 1.0
    assert coverage(40.0) == 0.0     # disjoint boxes, IoU 0.0


# -- 8. every field survives filtering with the same index ---------------


def test_score_threshold_keeps_all_fields_aligned():
    """The low-scoring prediction carries the *wrong* geometry.

    If the score mask were applied to some fields and not others, the surviving
    row would pick up that geometry and the reported errors would jump.
    """
    size = [0.02, 0.02, 0.02]
    sample, _ = _sample(
        centers=[[0.0, 0.0, 0.4], [5.0, 5.0, 5.0]],
        sigmas=[_sigma(size), _sigma([1.0, 1.0, 1.0])],
        gt_centers=[[0.0, 0.0, 0.4]], gt_sizes=[size],
        scores=torch.tensor([0.9, 0.01]))
    out = _run([(sample, None)], score_thr=0.2)
    assert out["prediction_count"] == 1.0
    assert out["envelope_iou_median"] == pytest.approx(1.0, abs=1e-6)


# -- 9/10. the centre term is what inflates Sigma ------------------------


def _kld_sigma_gradient(include_center):
    """Gradient of the KLD w.r.t. the predicted log-scale along the error axis.

    The prediction is shape-correct but displaced 40 mm in ``z``.  A negative
    gradient on ``log l33`` means the objective is actively paying the network
    to stretch the ellipsoid along the very axis it is wrong on.
    """
    log_scale = torch.zeros(1, 3, requires_grad=True)
    radius = 0.01
    predicted_cholesky = torch.diag_embed(
        radius * torch.exp(log_scale))
    target_cholesky = torch.diag_embed(
        torch.full((1, 3), radius))
    predicted_center = torch.tensor([[0.0, 0.0, 0.44]])
    target_center = torch.tensor([[0.0, 0.0, 0.40]])
    loss = Ellipsoid3DKLDLoss(include_center=include_center)(
        predicted_center, predicted_cholesky, target_center, target_cholesky)
    loss.backward()
    return log_scale.grad[0]


def test_centre_term_creates_pressure_to_inflate_the_error_axis():
    gradient = _kld_sigma_gradient(include_center=True)
    # d loss / d log l33 < 0: growing the z extent reduces the loss.
    assert gradient[2] < -1e-3
    # And it is specifically the error axis, not an isotropic effect.
    assert gradient[2] < gradient[0]
    assert gradient[2] < gradient[1]


def test_shape_only_kld_has_no_inflation_pressure():
    gradient = _kld_sigma_gradient(include_center=False)
    assert torch.allclose(gradient, torch.zeros(3), atol=1e-6), (
        "with include_center=False a shape-correct prediction must sit at a "
        f"stationary point, got {gradient.tolist()}")


# -- supporting helpers --------------------------------------------------


def test_projected_ellipse_layout_conversion():
    ellipse = torch.tensor([[3.0, 2.0, 10.0, 20.0, 0.5]])
    assert torch.allclose(
        projected_ellipse_to_rbox(ellipse),
        torch.tensor([[10.0, 20.0, 6.0, 4.0, 0.5]]))


def test_envelope_obb_round_trips_a_rotated_annotation():
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    size = np.array([0.04, 0.02, 0.03])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    sigma = gt_sigma_from_obb(transform, size)
    _, recovered, _ = sigma_to_envelope_obb(np.zeros(3), sigma)
    assert np.allclose(np.sort(recovered), np.sort(size), atol=1e-9)
