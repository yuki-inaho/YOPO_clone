from __future__ import annotations

import pytest
import torch

from tools.analysis_tools.expand_query_checkpoint import expand_query_embedding


def test_expand_query_embedding_preserves_existing_rows_and_is_deterministic():
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    first = expand_query_embedding(source, 10, seed=7)
    second = expand_query_embedding(source, 10, seed=7)

    assert first.shape == (10, 4)
    assert torch.equal(first[:6], source)
    assert torch.equal(first, second)
    assert not torch.equal(first[6:], source[:4])


def test_expand_query_embedding_rejects_shrink_and_invalid_rank():
    with pytest.raises(ValueError, match="cannot shrink"):
        expand_query_embedding(torch.zeros(6, 4), 5)
    with pytest.raises(ValueError, match="rank 2"):
        expand_query_embedding(torch.zeros(6), 8)
