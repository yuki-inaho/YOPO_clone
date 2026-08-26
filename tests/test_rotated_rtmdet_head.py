import math

import torch

from yopo.models.dense_heads.rotated_rtmdet_head import distance_angle_to_rbox
from yopo.models.task_modules.assigners.iou2d_calculator import RBboxOverlaps2D


def test_distance_angle_decoder_uses_rotated_local_offset():
    point = torch.tensor([[10.0, 20.0]])
    prediction = torch.tensor([[2.0, 3.0, 4.0, 5.0, 0.0]])

    actual = distance_angle_to_rbox(point, prediction)

    torch.testing.assert_close(
        actual,
        torch.tensor([[11.0, 21.0, 6.0, 8.0, 0.0]]),
    )


def test_distance_angle_decoder_normalizes_angle_to_le90():
    point = torch.zeros(1, 2)
    prediction = torch.tensor([[1.0, 1.0, 1.0, 1.0, math.pi]])

    actual = distance_angle_to_rbox(point, prediction)

    assert -math.pi / 2 <= actual[0, 4] < math.pi / 2
    torch.testing.assert_close(actual[0, 4], torch.tensor(0.0))


def test_rotated_overlap_calculator_keeps_five_dimensional_geometry():
    calculator = RBboxOverlaps2D()
    horizontal = torch.tensor([[20.0, 20.0, 12.0, 4.0, 0.0]])
    identical = horizontal.clone()
    perpendicular = torch.tensor(
        [[20.0, 20.0, 12.0, 4.0, math.pi / 2]])

    same_iou = calculator(horizontal, identical)
    rotated_iou = calculator(horizontal, perpendicular)

    torch.testing.assert_close(same_iou, torch.ones_like(same_iou))
    assert 0 < rotated_iou.item() < 1
