import torch

from yopo.models.layers.transformer.deformable_detr_layers import (
    AmpSafeMultiScaleDeformableAttention,
)


def test_bfloat16_input_uses_fp32_attention_and_returns_finite_bfloat16():
    attention = AmpSafeMultiScaleDeformableAttention(
        embed_dims=8,
        num_heads=2,
        num_levels=1,
        num_points=2,
        batch_first=True,
    )
    attention.init_weights()
    query = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    reference_points = torch.rand(2, 3, 1, 2, dtype=torch.bfloat16)

    output = attention(
        query=query,
        value=value,
        identity=query,
        reference_points=reference_points,
        spatial_shapes=torch.tensor([[2, 2]], dtype=torch.long),
        level_start_index=torch.tensor([0], dtype=torch.long),
    )

    assert output.shape == query.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
