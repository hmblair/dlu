"""Logging utilities for training.

This package provides pluggable logging backends for training loops,
including console progress bars and optional Weights & Biases integration.
"""
from __future__ import annotations

from .base import TrainingLogger
from .console import ConsoleProgress
from .wandb import WandbLogger, is_wandb_available

__all__ = [
    "TrainingLogger",
    "ConsoleProgress",
    "WandbLogger",
    "is_wandb_available",
]
