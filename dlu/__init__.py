"""DLU - Deep Learning Utilities for PyTorch.

A collection of utilities for deep learning research, including:
- Neural network modules (DenseNetwork, Attention, Transformer)
- Training loop utilities with pluggable logging
- Learning rate schedulers
- Data transformation and plotting utilities
"""
__version__ = "0.3.0"

# Neural network modules
from .modules import DenseNetwork, Attention, Transformer

# Schedulers
from .schedulers import LinearWarmupSqrtDecay

# Utilities
from .utils import params

# Transforms
from .transforms import normalize

# Plotting
from .plotting import plot_tensor

# Subpackages
from . import logging
from . import training

# Convenience aliases
norm = normalize
plot = plot_tensor

__all__ = [
    # Version
    "__version__",
    # Modules
    "DenseNetwork",
    "Attention",
    "Transformer",
    # Schedulers
    "LinearWarmupSqrtDecay",
    # Utilities
    "params",
    # Transforms
    "normalize",
    "norm",
    # Plotting
    "plot_tensor",
    "plot",
    # Subpackages
    "logging",
    "training",
]
