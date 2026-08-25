from __future__ import annotations

import numpy as np

from tools.analysis_tools.diagnose_rgbd_obb_rotation_correlation import (
    canonical_camera_rotation,
    gwd_distance,
    infer_obb_coordinate_scale,
    project_ellipsoid_gaussian,
    rbox_to_gaussian,
    summarize_observability,
)


def test_infer_obb_coordinate_scale_recovers_800_to_640_resize():
    obb_centers = np.array([[100.0, 200.0], [600.0, 400.0]])
    bbox_centers = obb_centers * 0.8

    scale, residuals = infer_obb_coordinate_scale(
        obb_centers, bbox_centers)

    np.testing.assert_allclose(scale, 0.8, atol=1e-12)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)


def test_rbox_gaussian_is_invariant_to_width_height_angle_swap():
    first = np.array([120.0, 80.0, 40.0, 20.0, 0.3])
    swapped = np.array([120.0, 80.0, 20.0, 40.0, 0.3 + np.pi / 2])

    first_xy, first_sigma = rbox_to_gaussian(first)
    swapped_xy, swapped_sigma = rbox_to_gaussian(swapped)

    np.testing.assert_allclose(first_xy, swapped_xy, atol=1e-12)
    np.testing.assert_allclose(first_sigma, swapped_sigma, atol=1e-10)
    np.testing.assert_allclose(gwd_distance(first, swapped), 0.0, atol=1e-6)


def test_projected_sphere_matches_perspective_analytic_solution():
    intrinsic = np.array(
        [[443.9066, 0.0, 321.3503], [0.0, 449.1953, 230.8687], [0, 0, 1]])
    translation = np.array([0.0, 0.0, 0.5])
    radius = 0.02
    size = np.full(3, 2 * radius)

    xy, sigma = project_ellipsoid_gaussian(
        translation, canonical_camera_rotation(), size, intrinsic)

    expected_radii = intrinsic[[0, 1], [0, 1]] * radius / np.sqrt(
        translation[2] ** 2 - radius**2)
    np.testing.assert_allclose(xy, intrinsic[:2, 2], atol=1e-8)
    np.testing.assert_allclose(np.diag(sigma), expected_radii**2, rtol=1e-10)
    np.testing.assert_allclose(sigma[0, 1], 0.0, atol=1e-10)


def test_observability_report_separates_low_and_high_anisotropy():
    paired = np.array([0.09, 0.08, 0.03, 0.02])
    canonical = np.array([0.10, 0.10, 0.09, 0.10])
    shuffled = np.array([0.10, 0.09, 0.12, 0.13])
    anisotropy = np.array([0.02, 0.08, 0.45, 0.55])

    report = summarize_observability(
        paired, canonical, shuffled, anisotropy,
        bin_edges=(0.0, 0.1, 0.4, 1.0),
    )

    assert report["anisotropy"]["count"] == 4
    assert report["correlation_with_shuffled_minus_paired"] > 0.9
    assert report["bins"][0]["count"] == 2
    assert report["bins"][2]["count"] == 2
    assert report["bins"][2]["paired_better_than_shuffled_fraction"] == 1.0
