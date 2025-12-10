"""Plotting utilities for visualizing tensors.

This module provides convenience functions for plotting PyTorch tensors
using matplotlib.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_tensor(
    ax: plt.Axes,
    y: torch.Tensor,
    start: float | None = None,
    end: float | None = None,
    **kwargs,
) -> None:
    """Plot a 1D tensor on matplotlib axes.

    Creates an x-axis from start to end with the same number of points
    as the tensor, then plots y values against it.

    Args:
        ax: Matplotlib axes to plot on.
        y: 1D tensor of values to plot.
        start: Start value for x-axis. Defaults to 0.
        end: End value for x-axis. Defaults to len(y) - 1.
        **kwargs: Additional arguments passed to ax.plot().

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> y = torch.sin(torch.linspace(0, 2 * 3.14159, 100))
        >>> plot_tensor(ax, y, start=0, end=2*3.14159)
        >>> plt.show()
    """
    start = start if start is not None else 0
    end = end if end is not None else y.size(0) - 1
    x = torch.linspace(start, end, y.size(0))

    # Move to CPU and convert to numpy for matplotlib
    ax.plot(x.numpy(), y.detach().cpu().numpy(), **kwargs)
