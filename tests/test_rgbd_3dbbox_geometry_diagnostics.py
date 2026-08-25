from __future__ import annotations

import numpy as np
import torch
from mmengine.logging.history_buffer import HistoryBuffer
from mmengine.structures import InstanceData

from tools.analysis_tools.diagnose_rgbd_3dbbox_geometry import (
    evaluate_geometry_contract,
    summarize_values,
)
from yopo.models.dense_pose_heads.dino_9d_center2d_posehead import (
    DINO9DCenter2DPoseHead,
)
from yopo.utils import register_mmengine_checkpoint_safe_globals


def _head() -> DINO9DCenter2DPoseHead:
    return DINO9DCenter2DPoseHead(
        num_classes=1,
        embed_dims=8,
        num_reg_fcs=1,
        num_pred_layer=1,
        train_cfg=None,
        use_cop_chain=False,
        use_log_z=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
    )


def test_matching_translation_uses_pixel_center_before_inverse_projection():
    head = _head()
    pred = head._build_matching_pred_instances(
        cls_score=torch.zeros(1, 1),
        bbox_pred=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        centers_2d_pred=torch.tensor([[0.5, 0.5]]),
        z_pred=torch.tensor([[np.log(2.0)]], dtype=torch.float32),
        rotation_pred=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        sizes_pred=torch.tensor([[0.2, 0.3, 0.4]]),
        img_meta=dict(
            img_shape=(480, 640),
            intrinsic=[100.0, 100.0, 320.0, 240.0],
        ),
    )

    assert torch.allclose(
        pred.translations,
        torch.tensor([[0.0, 0.0, 2.0]]),
        atol=1e-6,
    )


def test_geometry_contract_and_summary_are_json_safe():
    translations = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
    transforms = np.eye(4, dtype=np.float64)[None]
    transforms[0, :3, 3] = translations[0]
    centers = np.array([[320.0, 240.0]], dtype=np.float64)
    log_z = np.log(translations[:, 2:3])
    sizes = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)

    result = evaluate_geometry_contract(
        intrinsic=[100.0, 100.0, 320.0, 240.0],
        translations=translations,
        transforms=transforms,
        centers_2d=centers,
        log_z=log_z,
        sizes=sizes,
    )

    assert result["pass"] is True
    assert result["max_translation_T_error_m"] <= 1e-6
    assert result["max_center_projection_error_px"] <= 1e-6
    assert summarize_values(np.array([1.0, 2.0])) == {
        "count": 2,
        "finite_count": 2,
        "min": 1.0,
        "median": 1.5,
        "max": 2.0,
        "mean": 1.5,
        "std": 0.5,
    }


def test_geometry_contract_rejects_inconsistent_or_nonpositive_gt():
    transforms = np.eye(4, dtype=np.float64)[None]
    transforms[0, :3, 3] = [0.0, 0.0, -1.0]
    result = evaluate_geometry_contract(
        intrinsic=[100.0, 100.0, 320.0, 240.0],
        translations=np.array([[0.0, 0.0, -1.0]]),
        transforms=transforms,
        centers_2d=np.array([[0.0, 0.0]]),
        log_z=np.array([[np.nan]]),
        sizes=np.array([[0.2, 0.3, 0.4]]),
    )

    assert result["pass"] is False
    assert result["positive_translation_depth"] is False
    assert result["all_finite"] is False


def test_mmengine_checkpoint_metadata_is_weights_only_loadable(tmp_path):
    checkpoint = tmp_path / "mmengine.pth"
    history = HistoryBuffer()
    history.update(1.0)
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "message_hub": {
                "history": history,
                "numpy_scalar": np.float64(1.0),
            },
        },
        checkpoint,
    )

    register_mmengine_checkpoint_safe_globals()
    loaded = torch.load(checkpoint, weights_only=True)

    assert torch.equal(loaded["state_dict"]["weight"], torch.ones(1))
    assert isinstance(loaded["message_hub"]["history"], HistoryBuffer)
