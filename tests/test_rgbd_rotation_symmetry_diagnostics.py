from __future__ import annotations

import numpy as np

from tools.analysis_tools.diagnose_rgbd_rotation_symmetry import (
    axial_quotient_error_degrees,
    half_turn_quotient_error_degrees,
)


def _rotation(axis: int, angle: float) -> np.ndarray:
    matrix = np.eye(3)
    first, second = [index for index in range(3) if index != axis]
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[first, first] = matrix[second, second] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    return matrix


def test_continuous_axial_quotient_removes_rotation_about_same_local_axis():
    target = np.eye(3)[None]
    predicted = _rotation(1, np.pi / 2.0)[None]

    np.testing.assert_allclose(
        axial_quotient_error_degrees(predicted, target, axis=1), [0.0],
        atol=1e-6)
    np.testing.assert_allclose(
        axial_quotient_error_degrees(predicted, target, axis=0), [90.0],
        atol=1e-6)


def test_half_turn_quotient_only_identifies_180_degree_rotation():
    target = np.eye(3)[None]
    quarter = _rotation(1, np.pi / 2.0)[None]
    half = _rotation(1, np.pi)[None]

    np.testing.assert_allclose(
        half_turn_quotient_error_degrees(quarter, target, axis=1), [90.0],
        atol=1e-6)
    np.testing.assert_allclose(
        half_turn_quotient_error_degrees(half, target, axis=1), [0.0],
        atol=1e-6)
