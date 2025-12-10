"""Data transformation utilities.

This module provides functions for transforming and normalizing tensors,
with proper handling of NaN values.
"""
from __future__ import annotations

import torch


def normalize(
    x: torch.Tensor,
    clip: bool = False,
    use_minmax: bool = False,
) -> torch.Tensor:
    """Normalize a tensor, handling NaN values.

    Supports two normalization modes:
    - Z-score normalization (default): (x - mean) / std
    - Min-max normalization: (x - min) / (max - min)

    NaN values are ignored when computing statistics and preserved
    in the output.

    Args:
        x: Input tensor to normalize.
        clip: If True, clip values to [0, 1] before normalizing.
        use_minmax: If True, use min-max normalization.
            If False (default), use z-score normalization.

    Returns:
        Normalized tensor with same shape as input.

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0, float('nan'), 5.0])
        >>> normalize(x)  # z-score normalization
        tensor([-1.1832, -0.5071,  0.1690,     nan,  1.5213])
        >>> normalize(x, use_minmax=True)  # min-max normalization
        tensor([0.0000, 0.2500, 0.5000,    nan, 1.0000])
    """
    if clip:
        x = x.clip(0, 1)

    # Mask for valid (non-NaN) values
    valid_mask = ~x.isnan()

    if use_minmax:
        min_val = x[valid_mask].min()
        max_val = x[valid_mask].max()
        return (x - min_val) / (max_val - min_val)
    else:
        mean = x[valid_mask].mean()
        std = x[valid_mask].std()
        return (x - mean) / std
