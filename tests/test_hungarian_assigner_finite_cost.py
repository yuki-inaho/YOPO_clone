import pytest
import torch
from mmengine.structures import InstanceData

from yopo.models.task_modules.assigners.hungarian_assigner import HungarianAssigner
from yopo.registry import TASK_UTILS


@TASK_UTILS.register_module(force=True)
class _AlwaysNonFiniteCost:
    """A synthetic cost used only to verify the assigner's diagnostic gate."""

    def __call__(self, pred_instances, gt_instances, **kwargs):
        return torch.full(
            (len(pred_instances), len(gt_instances)),
            float("nan"),
            device=pred_instances.bboxes.device,
        )


def test_hungarian_assigner_names_nonfinite_cost_and_prediction_fields():
    """Assignment failures must identify the corrupt cost before SciPy sees it."""
    assigner = HungarianAssigner(match_costs=[dict(type="_AlwaysNonFiniteCost")])
    pred_instances = InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        translations=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    gt_instances = InstanceData(
        labels=torch.tensor([0]),
        bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    )

    with pytest.raises(
        FloatingPointError,
        match=r"_AlwaysNonFiniteCost.*pred_fields=.*bboxes.*translations",
    ):
        assigner.assign(pred_instances, gt_instances, img_meta={"img_shape": (1, 1)})
