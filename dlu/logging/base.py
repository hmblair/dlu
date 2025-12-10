"""Base protocol for training loggers.

This module defines the interface that all training loggers must implement.
"""
from __future__ import annotations

from typing import Protocol


class TrainingLogger(Protocol):
    """Protocol defining the interface for training loggers.

    Implement this protocol to create custom logging backends.
    The TrainingLoop class accepts any object implementing this protocol.

    Example:
        >>> class MyLogger:
        ...     def log_step(self, metrics: dict[str, float]) -> None:
        ...         print(f"Step metrics: {metrics}")
        ...
        ...     def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        ...         print(f"Epoch {epoch}: {metrics}")
        ...
        ...     def close(self) -> None:
        ...         pass
    """

    def log_step(self, metrics: dict[str, float]) -> None:
        """Log metrics for a single training step.

        Args:
            metrics: Dictionary of metric names to values.
        """
        ...

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics at the end of an epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of metric names to values.
        """
        ...

    def close(self) -> None:
        """Clean up resources when training is complete."""
        ...
