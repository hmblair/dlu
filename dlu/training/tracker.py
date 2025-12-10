"""Loss and metric tracking for training.

This module provides a simple class for tracking training metrics
such as loss values and computing running averages.
"""
from __future__ import annotations


class LossTracker:
    """Track loss values and compute running averages.

    A lightweight class for tracking training progress without
    any side effects or dependencies on external services.

    Args:
        name: Optional name prefix for metrics.

    Attributes:
        current_epoch: Current epoch number (0-indexed).
        current_step: Current step within the epoch.
        current_loss: Most recent loss value.
        average_loss: Running average loss for current epoch.

    Example:
        >>> tracker = LossTracker(name="train")
        >>> tracker.start_epoch()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     tracker.update(loss)
        >>> print(f"Average loss: {tracker.average_loss:.4f}")
    """

    def __init__(self: LossTracker, name: str = "") -> None:
        self.name = name
        self.current_epoch = -1
        self.current_step = -1
        self.current_loss = 0.0
        self.average_loss = 0.0

    def start_epoch(self) -> None:
        """Reset counters for a new epoch.

        Increments the epoch counter and resets step counter
        and average loss to initial values.
        """
        self.current_epoch += 1
        self.current_step = -1
        self.average_loss = 0.0

    def update(self, loss: float) -> None:
        """Record a loss value and update running average.

        Args:
            loss: The loss value to record.
        """
        self.current_step += 1
        self.current_loss = loss

        # Update running average
        self.average_loss = (
            self.current_step * self.average_loss + loss
        ) / (self.current_step + 1)

    @property
    def metrics(self) -> dict[str, float]:
        """Get current metrics as a dictionary.

        Returns:
            Dictionary with 'loss' and 'average_loss' keys,
            prefixed with name if provided.
        """
        prefix = f"{self.name}_" if self.name else ""
        return {
            f"{prefix}loss": self.current_loss,
            f"{prefix}average_loss": self.average_loss,
        }

    @property
    def epoch_metrics(self) -> dict[str, float]:
        """Get epoch-level metrics.

        Returns:
            Dictionary with average loss for logging at epoch end.
        """
        prefix = f"{self.name}_" if self.name else ""
        return {
            f"{prefix}epoch_loss": self.average_loss,
        }
