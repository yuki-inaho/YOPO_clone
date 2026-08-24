import numpy as np
import pytest

from tools.analysis_tools.render_rgbd_3dbbox_overlays import (
    intrinsic_matrix,
    project_corners,
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
