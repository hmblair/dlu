"""DLU - Deep Learning Utilities for PyTorch.

A collection of utilities for deep learning research, including:
- Neural network modules (DenseNetwork, RMSNorm, RoPE, SwiGLU, Transformer)
- Training loop utilities with pluggable logging
- Learning rate schedulers (warmup + cosine decay, warmup + sqrt decay)
- Data transformation and plotting utilities
"""
__version__ = "0.4.0"

# Neural network modules
from .modules import (
    DenseNetwork,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    MultiHeadAttention,
    TransformerBlock,
    Transformer,
)

# Schedulers
from .schedulers import LinearWarmupSqrtDecay, get_cosine_schedule_with_warmup

# Utilities
from .utils import params

# Transforms
from .transforms import normalize

# Plotting
from .plotting import plot_tensor

# Subpackages
from . import logging
from . import training
from . import lora

# Convenience aliases
norm = normalize
plot = plot_tensor

__all__ = [
    # Version
    "__version__",
    # Modules
    "DenseNetwork",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    "MultiHeadAttention",
    "TransformerBlock",
    "Transformer",
    # Schedulers
    "LinearWarmupSqrtDecay",
    "get_cosine_schedule_with_warmup",
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
    "lora",
]
