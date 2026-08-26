from __future__ import annotations

import pytest
import torch

from yopo.models.layers.transformer.oriented_contrastive_denoising import (
    BoxOnlyOCDNoise,
    generate_box_only_ocd_queries,
)
from yopo.models.layers.transformer.dino_layers import CdnQueryGenerator
from yopo.models.layers.transformer.utils import inverse_sigmoid
from yopo.registry import MODELS
from yopo.structures.bbox import bbox_cxcywh_to_xyxy


def _xyxy_boxes() -> torch.Tensor:
    return torch.tensor([
        [0.30, 0.35, 0.50, 0.55],
        [0.55, 0.40, 0.75, 0.70],
    ])


def _xyxy_from_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[:, :2]
    half_size = boxes[:, 2:] * 0.5
    return torch.cat((center - half_size, center + half_size), dim=-1)


def test_box_only_ocd_is_one_to_one_and_preserves_angle_metadata() -> None:
    boxes = _xyxy_boxes()
    labels = torch.tensor([3, 7])
    angles = torch.tensor([0.25, -1.0])

    queries = generate_box_only_ocd_queries(
        boxes,
        labels,
        box_format="xyxy",
        angles=angles,
        positive_noise_scale=0.05,
        negative_noise_scale=(0.2, 0.4),
        seed=17,
    )

    assert queries.boxes.shape == (4, 4)
    assert queries.labels.tolist() == [3, 3, 7, 7]
    torch.testing.assert_close(
        queries.angles, torch.tensor([0.25, 0.25, -1.0, -1.0]))
    assert queries.query_indices.tolist() == [0, 1, 2, 3]
    assert queries.source_box_indices.tolist() == [0, 0, 1, 1]
    assert queries.source_label_indices.tolist() == [0, 0, 1, 1]
    assert queries.positive_mask.tolist() == [True, False, True, False]
    assert queries.positive_query_indices.tolist() == [0, 2]
    assert queries.negative_query_indices.tolist() == [1, 3]
    assert queries.pair_query_indices.tolist() == [[0, 1], [2, 3]]

    source = boxes.repeat_interleave(2, dim=0)
    size = boxes[:, 2:] - boxes[:, :2]
    edge_scale = size.repeat(1, 2).repeat_interleave(2, dim=0)
    relative_noise = (queries.boxes - source).abs() / edge_scale
    assert bool((relative_noise[queries.positive_mask] <= 0.05).all())
    assert bool((relative_noise[~queries.positive_mask] >= 0.2).all())
    assert bool((relative_noise[~queries.positive_mask] <= 0.4).all())


def test_seed_is_reproducible_without_consuming_global_rng() -> None:
    boxes = _xyxy_boxes()
    labels = torch.tensor([0, 1])

    first = generate_box_only_ocd_queries(
        boxes, labels, box_format="xyxy", seed=123)
    second = generate_box_only_ocd_queries(
        boxes, labels, box_format="xyxy", seed=123)
    different = generate_box_only_ocd_queries(
        boxes, labels, box_format="xyxy", seed=124)

    torch.testing.assert_close(first.boxes, second.boxes)
    assert not torch.equal(first.boxes, different.boxes)

    torch.manual_seed(9)
    expected_next = torch.rand(4)
    torch.manual_seed(9)
    generate_box_only_ocd_queries(boxes, labels, box_format="xyxy", seed=5)
    actual_next = torch.rand(4)
    torch.testing.assert_close(actual_next, expected_next)


def test_cxcywh_outputs_are_clipped_and_repaired_to_min_size() -> None:
    boxes = torch.tensor([
        [0.990, 0.990, 0.002, 0.002],
        [0.010, 0.010, 0.004, 0.006],
    ])
    queries = generate_box_only_ocd_queries(
        boxes,
        torch.tensor([2, 4]),
        box_format="cxcywh",
        min_size=0.05,
        negative_noise_scale=(0.3, 0.45),
        seed=41,
    )

    output_xyxy = _xyxy_from_cxcywh(queries.boxes)
    assert bool((output_xyxy >= 0).all())
    assert bool((output_xyxy <= 1).all())
    output_size = output_xyxy[:, 2:] - output_xyxy[:, :2]
    assert bool((output_size >= 0.05 - 1e-7).all())


def test_empty_input_preserves_query_contract() -> None:
    queries = generate_box_only_ocd_queries(
        torch.empty((0, 4)),
        torch.empty((0,), dtype=torch.long),
        angles=torch.empty((0,)),
        seed=3,
    )

    assert queries.boxes.shape == (0, 4)
    assert queries.labels.shape == (0,)
    assert queries.angles.shape == (0,)
    assert queries.query_indices.shape == (0,)
    assert queries.pair_query_indices.shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"box_format": "xywh"}, "box_format"),
        ({"positive_noise_scale": 0.51}, "positive_noise_scale"),
        ({"negative_noise_scale": (0.04, 0.4)}, "negative noise"),
        ({"min_size": 0.0}, "min_size"),
    ],
)
def test_invalid_noise_contracts_are_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        generate_box_only_ocd_queries(
            _xyxy_boxes(), torch.tensor([0, 1]), **kwargs)


def test_seed_and_generator_are_mutually_exclusive() -> None:
    generator = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_box_only_ocd_queries(
            _xyxy_boxes(),
            torch.tensor([0, 1]),
            box_format="xyxy",
            seed=2,
            generator=generator,
        )


def test_nonnormalized_or_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="normalized"):
        generate_box_only_ocd_queries(
            torch.tensor([[0.1, 0.2, 1.2, 0.8]]),
            torch.tensor([0]),
            box_format="xyxy",
        )
    with pytest.raises(ValueError, match="angles"):
        generate_box_only_ocd_queries(
            _xyxy_boxes(),
            torch.tensor([0, 1]),
            box_format="xyxy",
            angles=torch.tensor([0.2]),
        )


def test_configurable_noise_strategy_preserves_dino_group_order() -> None:
    boxes = _xyxy_boxes()
    strategy = BoxOnlyOCDNoise(
        positive_noise_scale=0.05,
        negative_noise_scale=(0.2, 0.4),
    )
    torch.manual_seed(31)
    noisy = strategy(boxes, num_groups=2)

    assert noisy.shape == (8, 4)
    source_per_group = boxes.repeat(4, 1)
    edge_scale = (boxes[:, 2:] - boxes[:, :2]).repeat(1, 2)
    edge_scale_per_group = edge_scale.repeat(4, 1)
    relative = (noisy - source_per_group).abs() / edge_scale_per_group
    for group in range(2):
        start = group * 4
        assert bool((relative[start:start + 2] <= 0.05).all())
        assert bool((relative[start + 2:start + 4] >= 0.2).all())
        assert bool((relative[start + 2:start + 4] <= 0.4).all())


def test_cdn_generator_accepts_registry_box_noise_without_shape_changes() -> None:
    generator = CdnQueryGenerator(
        num_classes=1,
        embed_dims=8,
        num_matching_queries=16,
        label_noise_scale=0.5,
        box_noise=dict(
            type='BoxOnlyOCDNoise',
            positive_noise_scale=0.05,
            negative_noise_scale=(0.2, 0.4),
        ),
        group_cfg=dict(dynamic=False, num_groups=2),
    )
    torch.manual_seed(17)
    unactivated = generator.generate_dn_bbox_query(
        _xyxy_boxes(), num_groups=2)
    boxes = bbox_cxcywh_to_xyxy(unactivated.sigmoid())

    assert isinstance(generator.box_noise, BoxOnlyOCDNoise)
    assert unactivated.shape == (8, 4)
    assert torch.isfinite(unactivated).all()
    assert inverse_sigmoid(
        boxes.clamp(1e-3, 1 - 1e-3)).shape == unactivated.shape


def test_box_only_ocd_noise_is_registry_buildable() -> None:
    strategy = MODELS.build(dict(type='BoxOnlyOCDNoise'))
    assert isinstance(strategy, BoxOnlyOCDNoise)
