from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from yopo.evaluation.metrics import nocs_metric as nocs_metric_module
from yopo.evaluation.metrics.nocs_metric import (
    NOCSMetric,
    _float32_iou_upper_bound_can_match,
    compute_3d_iou,
    compute_3d_matches,
)


def _rt(rotation=None, translation=None, scale=1.0):
    transform = np.eye(4, dtype=np.float64)
    if rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    if translation is None:
        translation = np.zeros(3, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation) * scale
    transform[:3, 3] = translation
    return transform


def _iou(first_rt, second_rt, first_size, second_size, *, class_name="fruit"):
    return compute_3d_iou(
        first_rt,
        second_rt,
        np.asarray(first_size, dtype=np.float64),
        np.asarray(second_size, dtype=np.float64),
        1,
        class_name,
        class_name,
    )


def test_nocs_iou_is_one_for_identical_arbitrary_so3_boxes():
    rotation = Rotation.from_euler("xyz", [0.31, -0.72, 1.17]).as_matrix()
    transform = _rt(rotation, [1.2, -0.5, 3.4])

    assert _iou(transform, transform, [0.8, 1.7, 2.3], [0.8, 1.7, 2.3]) == 1.0


def test_nocs_iou_uses_oriented_volume_instead_of_corner_axis_reduction():
    size = [2.0, 4.0, 2.0]
    quarter_turn = Rotation.from_euler("z", np.pi / 2.0).as_matrix()

    iou = _iou(_rt(), _rt(quarter_turn), size, size)

    assert iou == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_nocs_iou_moves_uniform_rt_scale_into_full_side_lengths():
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    normalized_size = np.array([1.0, 2.0, 0.5])
    physical_size = normalized_size * 2.5

    iou = _iou(
        _rt(rotation, [0.1, 0.2, 3.0], scale=2.5),
        _rt(rotation, [0.1, 0.2, 3.0]),
        normalized_size,
        physical_size,
    )

    assert iou == 1.0


def test_nocs_iou_accepts_equivalent_negative_uniform_similarity_scale():
    rotation = Rotation.from_euler("xyz", [-0.4, 0.1, 0.7]).as_matrix()
    size = np.array([0.4, 0.8, 1.2])

    iou = _iou(
        _rt(rotation, scale=-2.0),
        _rt(rotation),
        size,
        size * 2.0,
    )

    assert iou == 1.0


def test_nocs_iou_keeps_declared_continuous_y_symmetry_policy():
    size = [2.0, 4.0, 1.0]
    y_turn = Rotation.from_euler("y", np.pi / 2.0).as_matrix()

    fruit_iou = _iou(_rt(), _rt(y_turn), size, size, class_name="fruit")
    bottle_iou = _iou(_rt(), _rt(y_turn), size, size, class_name="bottle")

    assert fruit_iou == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert bottle_iou == 1.0


def test_nocs_iou_preserves_missing_transform_sentinel():
    assert (
        compute_3d_iou(
            None,
            _rt(),
            np.ones(3),
            np.ones(3),
            1,
            "fruit",
            "fruit",
        )
        == -1
    )


def test_nocs_iou_rejects_nonuniform_rt_scale():
    invalid = _rt()
    invalid[:3, :3] = np.diag([1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="uniform scale"):
        _iou(invalid, _rt(), np.ones(3), np.ones(3))


def test_nocs_iou_projects_small_bfloat16_rotation_drift_to_so3():
    near_rotation = _rt()
    near_rotation[0, 0] += 1.5e-4

    assert _iou(near_rotation, _rt(), np.ones(3), np.ones(3)) > 0.999


def test_nocs_iou_polar_normalizes_measured_float32_rotation_drift():
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    drifted = rotation.copy()
    drifted[0, 1] += 2e-5

    assert _iou(_rt(drifted), _rt(rotation), np.ones(3), np.ones(3)) > 0.9999


@pytest.mark.parametrize(
    "size",
    [np.array([1.0, 0.0, 1.0]), np.array([1.0, -1.0, 1.0])],
)
def test_nocs_iou_maps_finite_nonpositive_prediction_size_to_zero(size):
    assert _iou(_rt(), _rt(), size, np.ones(3)) == 0.0


def test_nocs_iou_rejects_invalid_gt_size():
    with pytest.raises(ValueError, match="effective full sizes must be positive"):
        _iou(_rt(), _rt(), np.ones(3), np.array([1.0, 0.0, 1.0]))


def test_nocs_iou_does_not_hide_invalid_prediction_rt_behind_bad_size():
    invalid = _rt()
    invalid[:3, :3] = np.diag([1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="uniform scale"):
        _iou(invalid, _rt(), np.array([1.0, -1.0, 1.0]), np.ones(3))


def test_nocs_iou_rejects_nonfinite_prediction_size():
    with pytest.raises(ValueError, match="must be finite"):
        _iou(_rt(), _rt(), np.array([1.0, np.nan, 1.0]), np.ones(3))


def test_nocs_matching_reuses_decomposed_boxes_and_penalizes_invalid_prediction():
    gt_matches, pred_matches, overlaps, indices = compute_3d_matches(
        gt_class_ids=np.array([0]),
        gt_RTs=np.array([_rt()]),
        gt_scales=np.array([[2.0, 2.0, 2.0]]),
        gt_handle_visibility=np.array([1]),
        class_names=("fruit",),
        pred_boxes=np.array([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
        pred_class_ids=np.array([0, 0]),
        pred_scores=np.array([0.9, 0.8]),
        pred_RTs=np.array([_rt(), _rt()]),
        pred_scales=np.array([[2.0, 2.0, 2.0], [2.0, -1.0, 2.0]]),
        iou_3d_thresholds=[0.5],
    )

    assert overlaps[:, 0] == pytest.approx([1.0, 0.0])
    assert indices.tolist() == [0, 1]
    assert gt_matches.tolist() == [[0.0]]
    assert pred_matches.tolist() == [[0.0, -1.0]]


def test_pruned_matching_equals_all_exact_matching_on_random_boxes():
    rng = np.random.default_rng(20260826)
    num_pred = 9
    num_gt = 8
    pred_rotations = Rotation.random(num_pred, random_state=rng).as_matrix()
    gt_rotations = Rotation.random(num_gt, random_state=rng).as_matrix()
    pred_rts = np.stack(
        [_rt(rotation, rng.uniform(-0.8, 0.8, 3)) for rotation in pred_rotations]
    )
    gt_rts = np.stack(
        [_rt(rotation, rng.uniform(-0.8, 0.8, 3)) for rotation in gt_rotations]
    )
    pred_sizes = rng.uniform(0.2, 1.2, (num_pred, 3))
    gt_sizes = rng.uniform(0.2, 1.2, (num_gt, 3))
    pred_boxes = np.column_stack(
        (
            np.arange(num_pred) + 1.0,
            np.ones(num_pred),
            np.arange(num_pred) + 2.0,
            np.full(num_pred, 2.0),
        )
    )
    common = dict(
        gt_class_ids=np.zeros(num_gt, dtype=int),
        gt_RTs=gt_rts,
        gt_scales=gt_sizes,
        gt_handle_visibility=np.ones(num_gt, dtype=int),
        class_names=("fruit",),
        pred_boxes=pred_boxes,
        pred_class_ids=np.zeros(num_pred, dtype=int),
        pred_scores=np.linspace(0.99, 0.51, num_pred),
        pred_RTs=pred_rts,
        pred_scales=pred_sizes,
        iou_3d_thresholds=[0.1, 0.25, 0.5, 0.75],
    )

    exact = compute_3d_matches(**common, prune_below_iou_threshold=False)
    pruned = compute_3d_matches(**common, prune_below_iou_threshold=True)

    assert np.array_equal(pruned[0], exact[0])
    assert np.array_equal(pruned[1], exact[1])
    assert np.array_equal(pruned[3], exact[3])
    changed = pruned[2] != exact[2]
    assert np.all(exact[2][changed] < 0.1)
    assert np.all(pruned[2][changed] == 0.0)
    assert np.array_equal(pruned[2][exact[2] >= 0.1], exact[2][exact[2] >= 0.1])


def test_pruning_retains_exact_threshold_boundary_and_symmetry_pairs():
    threshold = 0.25
    shift = 2.0 * (1.0 - threshold) / (1.0 + threshold)
    common = dict(
        gt_class_ids=np.array([0]),
        gt_scales=np.array([[2.0, 2.0, 2.0]]),
        gt_handle_visibility=np.array([1]),
        class_names=("fruit",),
        pred_boxes=np.array([[1.0, 1.0, 2.0, 2.0]]),
        pred_class_ids=np.array([0]),
        pred_scores=np.array([0.9]),
        pred_RTs=np.array([_rt()]),
        pred_scales=np.array([[2.0, 2.0, 2.0]]),
        iou_3d_thresholds=[threshold],
    )
    exact = compute_3d_matches(
        gt_RTs=np.array([_rt(translation=[shift, 0.0, 0.0])]),
        **common,
    )
    pruned = compute_3d_matches(
        gt_RTs=np.array([_rt(translation=[shift, 0.0, 0.0])]),
        prune_below_iou_threshold=True,
        **common,
    )
    assert pruned[2][0, 0] == pytest.approx(exact[2][0, 0])
    assert np.array_equal(pruned[0], exact[0])
    assert np.array_equal(pruned[1], exact[1])

    anisotropic_size = np.array([[2.0, 4.0, 1.0]])
    symmetry_common = dict(common)
    symmetry_common.update(
        gt_RTs=np.array([_rt(Rotation.from_euler("y", np.pi / 2).as_matrix())]),
        gt_scales=anisotropic_size,
        pred_scales=anisotropic_size,
        class_names=("bottle",),
        iou_3d_thresholds=[0.5],
    )
    exact_symmetry = compute_3d_matches(**symmetry_common)
    pruned_symmetry = compute_3d_matches(
        **symmetry_common, prune_below_iou_threshold=True
    )
    assert exact_symmetry[2][0, 0] == 1.0
    assert np.array_equal(pruned_symmetry[2], exact_symmetry[2])
    assert np.array_equal(pruned_symmetry[0], exact_symmetry[0])


def test_two_phase_metric_switch_defaults_on_and_is_configurable():
    assert NOCSMetric().two_phase_3d_iou is True
    assert NOCSMetric(two_phase_3d_iou=False).two_phase_3d_iou is False
    with pytest.raises(TypeError, match="two_phase_3d_iou"):
        NOCSMetric(two_phase_3d_iou=1)


def test_pruning_stats_report_exact_phase_candidate_reduction():
    result = compute_3d_matches(
        gt_class_ids=np.array([0, 0]),
        gt_RTs=np.array([_rt(), _rt(translation=[10.0, 0.0, 0.0])]),
        gt_scales=np.full((2, 3), 2.0),
        gt_handle_visibility=np.ones(2, dtype=int),
        class_names=("fruit",),
        pred_boxes=np.array([[1.0, 1.0, 2.0, 2.0]]),
        pred_class_ids=np.array([0]),
        pred_scores=np.array([0.9]),
        pred_RTs=np.array([_rt()]),
        pred_scales=np.array([[2.0, 2.0, 2.0]]),
        iou_3d_thresholds=[0.1],
        prune_below_iou_threshold=True,
        return_pruning_stats=True,
    )

    assert result[4] == {"total_pairs": 2, "exact_candidates": 1}


def test_metric_worker_switch_preserves_end_to_end_pose_metrics():
    transform = _rt(translation=[0.0, 0.0, 1.0])
    preds = [
        {
            "labels": np.array([0]),
            "bboxes": np.array([[1.0, 1.0, 2.0, 2.0]]),
            "scores": np.array([0.9]),
            "T": np.array([transform]),
            "sizes": np.array([[1.0, 1.0, 1.0]]),
        }
    ]
    gts = [
        {
            "labels": np.array([0]),
            "T": np.array([transform]),
            "sizes": np.array([[1.0, 1.0, 1.0]]),
        }
    ]

    fast = NOCSMetric(two_phase_3d_iou=True).compute_independent_mAP(
        preds, gts, classes=("fruit",), num_workers=1
    )
    full = NOCSMetric(two_phase_3d_iou=False).compute_independent_mAP(
        preds, gts, classes=("fruit",), num_workers=1
    )

    assert fast == full


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("relation", [-1, 0, 1], ids=["below", "equal", "above"])
def test_float32_candidate_gate_is_conservative_at_rounding_midpoints(
    threshold, relation
):
    threshold_float32 = np.float32(threshold)
    lower = np.nextafter(threshold_float32, np.float32(-np.inf))
    upper = np.nextafter(threshold_float32, np.float32(np.inf))

    for adjacent in (lower, upper):
        midpoint = (float(threshold_float32) + float(adjacent)) / 2.0
        if relation < 0:
            upper_bound = np.nextafter(midpoint, -np.inf)
        elif relation > 0:
            upper_bound = np.nextafter(midpoint, np.inf)
        else:
            upper_bound = midpoint

        rounded = np.float32(upper_bound)
        guarded = np.nextafter(rounded, np.float32(np.inf))
        expected = float(guarded) > threshold
        actual = bool(
            _float32_iou_upper_bound_can_match(np.array([[upper_bound]]), threshold)[
                0, 0
            ]
        )

        assert actual is expected
        if float(rounded) > threshold:
            assert actual is True


def test_two_phase_retains_concrete_float32_threshold_rounding_match(monkeypatch):
    threshold = 0.1
    exact_iou_target = threshold - 2e-9
    shift = 2.0 * (1.0 - exact_iou_target) / (1.0 + exact_iou_target)
    common = dict(
        gt_class_ids=np.array([0]),
        gt_RTs=np.array([_rt(translation=[shift, 0.0, 0.0])]),
        gt_scales=np.array([[2.0, 2.0, 2.0]]),
        gt_handle_visibility=np.array([1]),
        class_names=("fruit",),
        pred_boxes=np.array([[1.0, 1.0, 2.0, 2.0]]),
        pred_class_ids=np.array([0]),
        pred_scores=np.array([0.9]),
        pred_RTs=np.array([_rt()]),
        pred_scales=np.array([[2.0, 2.0, 2.0]]),
        iou_3d_thresholds=[threshold],
    )
    full = compute_3d_matches(**common)
    assert float(full[2][0, 0]) > threshold
    assert full[1][0, 0] == 0.0

    monkeypatch.setattr(
        nocs_metric_module,
        "pairwise_oriented_box_iou_upper_bound_from_boxes_3d",
        lambda *args, **kwargs: np.array([[exact_iou_target]], dtype=np.float64),
    )
    fast = compute_3d_matches(**common, prune_below_iou_threshold=True)

    assert np.array_equal(fast[0], full[0])
    assert np.array_equal(fast[1], full[1])
    assert np.array_equal(fast[2], full[2])


def test_zero_padding_is_removed_with_one_aligned_mask_and_original_indices():
    padded_invalid_rt = np.zeros((4, 4), dtype=np.float64)
    result = compute_3d_matches(
        gt_class_ids=np.array([0]),
        gt_RTs=np.array([_rt()]),
        gt_scales=np.array([[2.0, 2.0, 2.0]]),
        gt_handle_visibility=np.array([1]),
        class_names=("fruit",),
        pred_boxes=np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        ),
        pred_class_ids=np.array([0, 0, 0]),
        pred_scores=np.array([0.5, 0.99, 0.9]),
        pred_RTs=np.array([_rt(), padded_invalid_rt, _rt(translation=[10, 0, 0])]),
        pred_scales=np.array(
            [[2.0, 2.0, 2.0], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]]
        ),
        iou_3d_thresholds=[0.1],
        prune_below_iou_threshold=True,
        return_pruning_stats=True,
    )

    assert result[2].shape == (2, 1)
    assert result[3].tolist() == [2, 0]
    assert result[4]["total_pairs"] == 2
