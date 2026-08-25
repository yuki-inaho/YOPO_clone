from __future__ import annotations

import numpy as np

from tools.analysis_tools.diagnose_predicted_obb_alignment import (
    normalized_rboxes_to_compact_gaussians,
    summarize_obb_alignment,
)


def test_normalized_rbox_gaussian_respects_source_aspect_ratio():
    rboxes = np.array([
        [400.0, 300.0, 200.0, 60.0, 0.0],
        [400.0, 300.0, 200.0, 60.0, np.pi / 2.0],
    ])

    compact = normalized_rboxes_to_compact_gaussians(rboxes)

    assert compact.shape == (2, 5)
    assert compact[0, 2] > compact[0, 4]
    assert compact[1, 2] < compact[1, 4]
    np.testing.assert_allclose(compact[:, :2], 0.0)


def test_perfect_predicted_obb_has_unit_alignment_and_zero_axis_error():
    rboxes = np.array([
        [100.0, 100.0, 120.0, 30.0, 0.0],
        [200.0, 200.0, 100.0, 50.0, 0.3],
        [300.0, 300.0, 80.0, 60.0, -0.7],
    ])
    target = normalized_rboxes_to_compact_gaussians(rboxes)

    report = summarize_obb_alignment(target.copy(), target)

    assert report["count"] == 3
    assert report["anisotropy_correlation"] == 1.0
    assert report["weighted_double_angle_alignment"] == 1.0
    assert report["axis_error_degrees"]["max"] < 1e-6
    assert report["normalized_gwd"]["max"] < 1e-6


def test_perpendicular_predicted_axis_has_negative_spin2_alignment():
    target = normalized_rboxes_to_compact_gaussians(np.array([
        [100.0, 100.0, 120.0, 30.0, 0.0],
    ]))
    predicted = normalized_rboxes_to_compact_gaussians(np.array([
        [100.0, 100.0, 120.0, 30.0, np.pi / 2.0],
    ]))

    report = summarize_obb_alignment(predicted, target)

    assert report["weighted_double_angle_alignment"] < -0.999999
    assert report["axis_error_degrees"]["median"] > 89.999
