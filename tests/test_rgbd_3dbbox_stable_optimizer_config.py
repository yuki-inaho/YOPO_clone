from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"


def test_rgbd_3dbbox_transfer_uses_conservative_detr_learning_rate():
    """Random 3D heads must not use the inherited 2.5e-3 scratch LR."""
    cfg = Config.fromfile(CONFIG_PATH)

    assert cfg.optim_wrapper.optimizer.type == "AdamWScheduleFreeOptimizer"
    assert cfg.optim_wrapper.optimizer.lr <= 2e-4
