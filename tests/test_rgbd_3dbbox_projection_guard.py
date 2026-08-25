import numpy as np
import pytest
import cv2

from tools.analysis_tools.render_rgbd_3dbbox_overlays import (
    intrinsic_matrix,
    project_corners,
    query_candidates,
    render_many,
    validate_options,
)


def test_projection_guard_rejects_invalid_camera_geometry():
    with pytest.raises(ValueError, match="intrinsic"):
        intrinsic_matrix([1.0, 2.0, 3.0])

    intrinsic = intrinsic_matrix([100.0, 100.0, 20.0, 20.0])
    with pytest.raises(ValueError, match="non-positive camera depth"):
        project_corners(np.zeros((8, 3)), intrinsic)

    invalid = np.ones((8, 3))
    invalid[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        project_corners(invalid, intrinsic)


def test_all_query_selection_threshold_sorting_and_render(tmp_path):
    image_path = tmp_path / "synthetic.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[:, 2, 3] = 1.0
    record = {
        "img_path": str(image_path),
        "intrinsic": [100.0, 100.0, 32.0, 24.0],
        "pred_instances": {
            "scores": np.array([0.3, 0.8, 0.5]),
            "T": transforms,
            "sizes": np.full((3, 3), 0.2),
        },
    }

    queries = query_candidates(record, index=4, threshold=0.45)

    assert [query["query_index"] for query in queries] == [1, 2]
    assert [query["score"] for query in queries] == [0.8, 0.5]
    assert all(query["record_index"] == 4 for query in queries)
    rendered = render_many(image, queries, threshold=0.45)
    assert rendered.shape == image.shape
    assert np.any(rendered != image)

    record["pred_instances"]["T"] = transforms[:2]
    with pytest.raises(ValueError, match="query geometry mismatch"):
        query_candidates(record, index=4, threshold=None)


def test_all_query_options_and_nonfinite_scores_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="num-images"):
        validate_options(0, all_queries=True, score_threshold=None)
    with pytest.raises(ValueError, match="requires --all-queries"):
        validate_options(1, all_queries=False, score_threshold=0.5)
    for threshold in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_options(1, all_queries=True, score_threshold=threshold)

    image_path = tmp_path / "synthetic.png"
    assert cv2.imwrite(
        str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    record = {
        "img_path": str(image_path),
        "intrinsic": [10.0, 10.0, 8.0, 8.0],
        "pred_instances": {
            "scores": np.array([np.nan]),
            "T": np.eye(4)[None],
            "sizes": np.ones((1, 3)),
        },
    }
    with pytest.raises(FloatingPointError, match="invalid scores"):
        query_candidates(record, index=0, threshold=None)
