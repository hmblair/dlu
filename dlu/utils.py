"""Utility functions for deep learning.

This module provides helper functions for common deep learning tasks
such as counting model parameters.
"""

from __future__ import annotations

import torch.nn as nn


def params(module: nn.Module) -> int:
    """Count the total number of parameters in a module.

    Args:
        module: PyTorch module to count parameters for.

    Returns:
        Total number of parameters (trainable and non-trainable).

    Example:
        >>> model = nn.Linear(10, 5)
        >>> params(model)
        55
    """
    return sum(x.numel() for x in module.parameters())
