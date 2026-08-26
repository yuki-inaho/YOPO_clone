from __future__ import annotations

import pytest
import torch

from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    aligned_iou_quality_targets,
)
from yopo.models.dense_pose_heads.matchability_quality import (
    MatchabilityQualityPolicy,
)
from yopo.models.losses.projected_ellipsoid_loss import (
    gaussian_wasserstein_distance,
)
from yopo.registry import MODELS


def _hbb_inputs(*, requires_grad: bool = False):
    predictions = torch.tensor(
        [[0.5, 0.5, 0.4, 0.4],
         [0.2, 0.2, 0.1, 0.1],
         [0.8, 0.8, 0.1, 0.1]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    targets = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2],
         [0.8, 0.8, 0.1, 0.1],
         [0.8, 0.8, 0.1, 0.1]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    labels = torch.tensor([0, 0, 1])
    return labels, predictions, targets


def _compact_gaussians(*, requires_grad: bool = False):
    predictions = torch.tensor(
        [[0.5, 0.5, 0.04, 0.0, 0.01],
         [0.2, 0.2, 0.01, 0.0, 0.02],
         [float('nan'), 0.0, -1.0, 0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    targets = torch.tensor(
        [[0.5, 0.5, 0.04, 0.0, 0.01],
         [0.8, 0.8, 0.02, 0.0, 0.01],
         [float('nan'), 0.0, -1.0, 0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    return predictions, targets


def test_hbb_source_reuses_aligned_iou_helper_and_detaches_geometry():
    labels, predictions, targets = _hbb_inputs(requires_grad=True)
    policy = MatchabilityQualityPolicy(
        num_classes=1, source='hbb_iou')

    actual = policy(labels, predictions, targets)
    expected = aligned_iou_quality_targets(
        labels, predictions, targets, num_classes=1)

    torch.testing.assert_close(actual, expected)
    assert actual.tolist() == pytest.approx([0.25, 0.0, 0.0])
    assert actual.requires_grad is False


def test_obb_source_is_complement_of_existing_bounded_gwd():
    labels, bbox_predictions, bbox_targets = _hbb_inputs()
    obb_predictions, obb_targets = _compact_gaussians()
    policy = MatchabilityQualityPolicy(
        num_classes=1,
        source='obb_gwd',
        normalize=True,
        include_center=False,
        tau=1.25,
    )

    actual = policy(
        labels, bbox_predictions, bbox_targets,
        obb_predictions, obb_targets)
    distance, valid = gaussian_wasserstein_distance(
        obb_predictions[:2], obb_targets[:2],
        normalize=True, include_center=False)
    expected = torch.zeros(3, dtype=torch.float64)
    expected[:2] = torch.where(valid, 1.0 / (1.25 + distance), 0.0)

    torch.testing.assert_close(actual, expected)
    assert torch.all((actual >= 0.0) & (actual <= 1.0))
    assert actual[2] == 0.0


def test_blend_is_configurable_convex_combination():
    labels, bbox_predictions, bbox_targets = _hbb_inputs()
    obb_predictions, obb_targets = _compact_gaussians()
    hbb = MatchabilityQualityPolicy(num_classes=1, source='hbb_iou')(
        labels, bbox_predictions, bbox_targets)
    obb = MatchabilityQualityPolicy(num_classes=1, source='obb_gwd')(
        labels, bbox_predictions, bbox_targets,
        obb_predictions, obb_targets)
    blend = MatchabilityQualityPolicy(
        num_classes=1, source='blend', obb_weight=0.3)(
            labels, bbox_predictions, bbox_targets,
            obb_predictions, obb_targets)

    torch.testing.assert_close(blend, 0.7 * hbb + 0.3 * obb)


@pytest.mark.parametrize('source', ['obb_gwd', 'blend'])
def test_missing_obb_requires_explicit_fallback(source):
    labels, predictions, targets = _hbb_inputs()
    with pytest.raises(ValueError, match='requires compact Gaussian'):
        MatchabilityQualityPolicy(num_classes=1, source=source)(
            labels, predictions, targets)

    hbb = MatchabilityQualityPolicy(num_classes=1, source='hbb_iou')(
        labels, predictions, targets)
    fallback = MatchabilityQualityPolicy(
        num_classes=1,
        source=source,
        missing_obb='hbb_iou',
    )(labels, predictions, targets)
    torch.testing.assert_close(fallback, hbb)


def test_zero_weight_blend_does_not_require_unused_obb_inputs():
    labels, predictions, targets = _hbb_inputs()
    hbb = MatchabilityQualityPolicy(num_classes=1, source='hbb_iou')(
        labels, predictions, targets)
    blend = MatchabilityQualityPolicy(
        num_classes=1, source='blend', obb_weight=0.0)(
            labels, predictions, targets)
    torch.testing.assert_close(blend, hbb)


def test_invalid_positive_gaussian_can_fail_or_be_zeroed():
    labels, bbox_predictions, bbox_targets = _hbb_inputs()
    labels = torch.tensor([0, 1, 1])
    obb_predictions, obb_targets = _compact_gaussians()
    obb_predictions = obb_predictions.clone()
    obb_predictions[0, 2] = -1.0

    with pytest.raises(RuntimeError, match='invalid positive compact'):
        MatchabilityQualityPolicy(
            num_classes=1, source='obb_gwd', fail_on_invalid=True)(
                labels, bbox_predictions, bbox_targets,
                obb_predictions, obb_targets)

    quality = MatchabilityQualityPolicy(
        num_classes=1, source='obb_gwd', fail_on_invalid=False)(
            labels, bbox_predictions, bbox_targets,
            obb_predictions, obb_targets)
    assert quality.tolist() == [0.0, 0.0, 0.0]
    assert torch.isfinite(quality).all()


def test_invalid_background_gaussian_is_ignored():
    labels, bbox_predictions, bbox_targets = _hbb_inputs()
    obb_predictions, obb_targets = _compact_gaussians()

    quality = MatchabilityQualityPolicy(
        num_classes=1, source='obb_gwd', fail_on_invalid=True)(
            labels, bbox_predictions, bbox_targets,
            obb_predictions, obb_targets)

    assert quality[2] == 0.0
    assert torch.isfinite(quality).all()


def test_all_geometry_inputs_are_detached():
    labels, bbox_predictions, bbox_targets = _hbb_inputs(requires_grad=True)
    obb_predictions, obb_targets = _compact_gaussians(requires_grad=True)
    quality = MatchabilityQualityPolicy(
        num_classes=1, source='blend', obb_weight=0.5)(
            labels, bbox_predictions, bbox_targets,
            obb_predictions, obb_targets)

    assert quality.requires_grad is False
    assert bbox_predictions.grad is None
    assert bbox_targets.grad is None
    assert obb_predictions.grad is None
    assert obb_targets.grad is None


def test_invalid_positive_hbb_fails_loudly_or_is_zeroed():
    labels, predictions, targets = _hbb_inputs()
    predictions = predictions.clone()
    predictions[0, 2] = float('nan')

    with pytest.raises(RuntimeError, match='invalid positive HBB'):
        MatchabilityQualityPolicy(
            num_classes=1, source='hbb_iou', fail_on_invalid=True)(
                labels, predictions, targets)

    quality = MatchabilityQualityPolicy(
        num_classes=1, source='hbb_iou', fail_on_invalid=False)(
            labels, predictions, targets)
    assert quality[0] == 0.0
    assert torch.isfinite(quality).all()


def test_low_precision_geometry_is_promoted_and_quality_is_bounded():
    labels, bbox_predictions, bbox_targets = _hbb_inputs()
    obb_predictions, obb_targets = _compact_gaussians()
    quality = MatchabilityQualityPolicy(
        num_classes=1, source='blend', obb_weight=0.5)(
            labels,
            bbox_predictions.to(torch.bfloat16),
            bbox_targets.to(torch.bfloat16),
            obb_predictions.to(torch.bfloat16),
            obb_targets.to(torch.bfloat16),
        )

    assert quality.dtype == torch.float32
    assert torch.isfinite(quality).all()
    assert torch.all((quality >= 0.0) & (quality <= 1.0))


def test_registry_build_and_configuration_validation():
    policy = MODELS.build(
        dict(
            type='MatchabilityQualityPolicy',
            num_classes=2,
            source='blend',
            obb_weight=0.7,
            normalize=False,
            include_center=True,
            tau=2.0,
            missing_obb='hbb_iou',
            fail_on_invalid=False,
        ))
    assert isinstance(policy, MatchabilityQualityPolicy)
    assert policy.source == 'blend'
    assert policy.obb_weight == 0.7
    assert policy.normalize is False
    assert policy.include_center is True
    assert policy.tau == 2.0

    with pytest.raises(ValueError, match='source'):
        MatchabilityQualityPolicy(num_classes=1, source='unknown')
    with pytest.raises(ValueError, match='obb_weight'):
        MatchabilityQualityPolicy(num_classes=1, obb_weight=1.1)
    with pytest.raises(ValueError, match='tau'):
        MatchabilityQualityPolicy(num_classes=1, tau=0.5)
    with pytest.raises(ValueError, match='missing_obb'):
        MatchabilityQualityPolicy(num_classes=1, missing_obb='silent')
