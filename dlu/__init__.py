"""DLU - Deep Learning Utilities for PyTorch.

A collection of utilities for deep learning research, including:
- Neural network modules (DenseNetwork, RMSNorm, RoPE, SwiGLU, Transformer)
- Training loop utilities with pluggable logging
- Learning rate schedulers (warmup + cosine decay, warmup + sqrt decay)
- Data transformation and plotting utilities
"""

__version__ = "0.4.0"

# Neural network modules
# Subpackages
from . import logging, lora, training
from .modules import (
    DenseNetwork,
    MultiHeadAttention,
    RadialBasisFunctions,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    Transformer,
    TransformerBlock,
)

# Plotting
from .plotting import plot_tensor

# Schedulers
from .schedulers import LinearWarmupSqrtDecay, get_cosine_schedule_with_warmup

# Transforms
from .transforms import normalize

# Utilities
from .utils import params

# Convenience aliases
norm = normalize
plot = plot_tensor

__all__ = [
    # Version
    "__version__",
    # Modules
    "DenseNetwork",
    "RadialBasisFunctions",
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
