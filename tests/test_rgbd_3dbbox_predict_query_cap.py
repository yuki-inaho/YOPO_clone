from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py"


def test_prediction_cap_does_not_exceed_available_class_query_scores():
    """The evaluator must not request more scores than the detector produces."""
    cfg = Config.fromfile(CONFIG_PATH)

    available_scores = cfg.model.num_queries * cfg.model.bbox_head.num_classes
    assert cfg.model.test_cfg.max_per_img <= available_scores
