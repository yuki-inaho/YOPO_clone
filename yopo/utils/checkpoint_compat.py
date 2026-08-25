"""Narrow PyTorch 2.6+ safe-loading compatibility for MMEngine checkpoints."""

from __future__ import annotations

import numpy as np
import torch
from mmengine.logging.history_buffer import HistoryBuffer
from numpy.core.multiarray import _reconstruct, scalar


def register_mmengine_checkpoint_safe_globals() -> None:
    """Allow the metadata types emitted by MMEngine's checkpoint hook.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default.
    MMEngine 0.10 checkpoint metadata contains HistoryBuffer and NumPy scalar
    values, so older checkpoints need these exact non-executable container and
    dtype types allowlisted before MMEngine calls ``torch.load``.
    """
    torch.serialization.add_safe_globals([
        HistoryBuffer,
        _reconstruct,
        scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        getattr,
    ])
