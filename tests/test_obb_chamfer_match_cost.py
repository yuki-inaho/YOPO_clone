import math

import pytest
import torch
from mmengine.structures import InstanceData

from yopo.models.task_modules.assigners.hungarian_assigner import HungarianAssigner
from yopo.models.task_modules.assigners.match_cost import OBBChamferCost
from yopo.registry import TASK_UTILS
from yopo.structures.bbox import RotatedBoxes


def _instances(pred_bboxes, gt_bboxes):
    return (
        InstanceData(bboxes=torch.tensor(pred_bboxes, dtype=torch.float32)),
        InstanceData(bboxes=torch.tensor(gt_bboxes, dtype=torch.float32)),
    )


def test_paper_squared_and_code_unsquared_are_distinct_and_normalized():
    pred, gt = _instances(
        [[5.0, 10.0, 2.0, 2.0, 0.0]],
        [[6.0, 10.0, 2.0, 2.0, 0.0]],
    )
    img_meta = {"img_shape": (20, 10)}

    squared = OBBChamferCost(distance_mode="paper_squared")(
        pred, gt, img_meta=img_meta
    )
    unsquared = OBBChamferCost(distance_mode="code_unsquared")(
        pred, gt, img_meta=img_meta
    )

    # Every corresponding corner is displaced by 1 / image_width = 0.1.
    # Bidirectional Chamfer sums the two directed means.
    torch.testing.assert_close(squared, torch.tensor([[0.02]]))
    torch.testing.assert_close(unsquared, torch.tensor([[0.2]]))


def test_weight_is_applied_after_chamfer_reduction():
    pred, gt = _instances(
        [[5.0, 10.0, 2.0, 2.0, 0.0]],
        [[6.0, 10.0, 2.0, 2.0, 0.0]],
    )

    cost = OBBChamferCost(
        distance_mode="paper_squared", weight=3.0
    )(pred, gt, img_meta={"img_shape": (20, 10)})

    torch.testing.assert_close(cost, torch.tensor([[0.06]]))


def test_corners_are_normalized_after_rotation_on_rectangular_images():
    pred, gt = _instances(
        [[5.0, 10.0, 2.0, 2.0, 0.0]],
        [[5.0, 10.0, 2.0, 2.0, math.pi / 2]],
    )

    cost = OBBChamferCost(equivalence_mode="none")(
        pred, gt, img_meta={"img_shape": (20, 10)}
    )

    # A square rotated by pi / 2 is the same point set. Dividing w/h before
    # rotation would incorrectly turn this into two different rectangles when
    # image height and width differ.
    torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("equivalence_mode", "equivalent_box"),
    [
        ("pi", [8.0, 7.0, 4.0, 2.0, 0.3 + math.pi]),
        ("pi_and_swap", [8.0, 7.0, 2.0, 4.0, 0.3 + math.pi / 2]),
    ],
)
def test_equivalent_obb_encodings_have_zero_cost(
    equivalence_mode, equivalent_box
):
    pred, gt = _instances(
        [[8.0, 7.0, 4.0, 2.0, 0.3]],
        [equivalent_box],
    )

    cost = OBBChamferCost(equivalence_mode=equivalence_mode)(
        pred, gt, img_meta={"img_shape": (20, 30)}
    )

    torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1e-12, rtol=0)


def test_equivalence_mode_controls_candidate_enumeration():
    boxes = torch.tensor([[8.0, 7.0, 4.0, 2.0, 0.3]])

    assert OBBChamferCost(equivalence_mode="none")._equivalent_boxes(
        boxes
    ).shape == (1, 1, 5)
    assert OBBChamferCost(equivalence_mode="pi")._equivalent_boxes(
        boxes
    ).shape == (2, 1, 5)
    assert OBBChamferCost(equivalence_mode="pi_and_swap")._equivalent_boxes(
        boxes
    ).shape == (4, 1, 5)


@pytest.mark.parametrize("bad_side", ["prediction", "ground truth"])
def test_nonfinite_input_fails_at_the_cost_boundary(bad_side):
    pred_box = [5.0, 10.0, 2.0, 2.0, 0.0]
    gt_box = [6.0, 10.0, 2.0, 2.0, 0.0]
    if bad_side == "prediction":
        pred_box[0] = float("nan")
    else:
        gt_box[4] = float("inf")
    pred, gt = _instances([pred_box], [gt_box])

    with pytest.raises(FloatingPointError, match=f"non-finite {bad_side}"):
        OBBChamferCost()(pred, gt, img_meta={"img_shape": (20, 10)})


def test_finite_input_that_overflows_the_cost_is_reported():
    pred, gt = _instances(
        [[0.0, 0.0, 1e30, 1e30, 0.0]],
        [[0.0, 0.0, 1.0, 1.0, 0.0]],
    )

    with pytest.raises(FloatingPointError, match="produced a non-finite cost"):
        OBBChamferCost()(pred, gt, img_meta={"img_shape": (1, 1)})


@pytest.mark.parametrize(
    "img_meta",
    [None, {}, {"img_shape": (0, 10)}, {"img_shape": (20, float("nan"))}],
)
def test_invalid_image_shape_is_rejected(img_meta):
    pred, gt = _instances(
        [[5.0, 10.0, 2.0, 2.0, 0.0]],
        [[6.0, 10.0, 2.0, 2.0, 0.0]],
    )

    with pytest.raises(ValueError, match="img_shape"):
        OBBChamferCost()(pred, gt, img_meta=img_meta)


def test_empty_input_preserves_hungarian_cost_shape():
    pred, gt = _instances([], [[6.0, 10.0, 2.0, 2.0, 0.0]])
    pred.bboxes = pred.bboxes.reshape(0, 5)

    cost = OBBChamferCost()(pred, gt, img_meta={"img_shape": (20, 10)})

    assert cost.shape == (0, 1)
    assert cost.dtype == torch.float32


def test_empty_ground_truth_preserves_hungarian_cost_shape():
    pred, gt = _instances([[5.0, 10.0, 2.0, 2.0, 0.0]], [])
    gt.bboxes = gt.bboxes.reshape(0, 5)

    cost = OBBChamferCost(equivalence_mode="pi_and_swap")(
        pred, gt, img_meta={"img_shape": (20, 10)}
    )

    assert cost.shape == (1, 0)
    assert cost.dtype == torch.float32


def test_ground_truth_accepts_the_existing_rotated_boxes_contract():
    pred, _ = _instances(
        [[5.0, 10.0, 2.0, 4.0, 0.2 + math.pi / 2]],
        [[5.0, 10.0, 4.0, 2.0, 0.2]],
    )
    gt = InstanceData(
        bboxes=RotatedBoxes([[5.0, 10.0, 4.0, 2.0, 0.2]])
    )

    cost = OBBChamferCost(equivalence_mode="pi_and_swap")(
        pred, gt, img_meta={"img_shape": (20, 10)}
    )

    torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1e-12, rtol=0)


def test_cost_is_available_through_the_task_utils_registry():
    cost = TASK_UTILS.build(
        dict(
            type="OBBChamferCost",
            distance_mode="code_unsquared",
            equivalence_mode="pi_and_swap",
            weight=2.0,
        )
    )

    assert isinstance(cost, OBBChamferCost)


def test_cost_integrates_with_hungarian_assigner():
    assigner = HungarianAssigner(
        match_costs=[
            dict(
                type="OBBChamferCost",
                distance_mode="paper_squared",
                equivalence_mode="pi_and_swap",
            )
        ]
    )
    pred = InstanceData(
        bboxes=torch.tensor(
            [
                [20.0, 10.0, 4.0, 2.0, 0.2],
                [5.0, 10.0, 2.0, 4.0, 0.2 + math.pi / 2],
            ]
        )
    )
    gt = InstanceData(
        bboxes=torch.tensor(
            [
                [5.0, 10.0, 4.0, 2.0, 0.2],
                [20.0, 10.0, 4.0, 2.0, 0.2 + math.pi],
            ]
        ),
        labels=torch.tensor([3, 7]),
    )

    result = assigner.assign(pred, gt, img_meta={"img_shape": (20, 30)})

    assert result.gt_inds.tolist() == [2, 1]
    assert result.labels.tolist() == [7, 3]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("distance_mode", "unknown"),
        ("equivalence_mode", "unknown"),
    ],
)
def test_invalid_modes_are_rejected(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        OBBChamferCost(**{keyword: value})
