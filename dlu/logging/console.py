"""Console-based progress display using tqdm.

This module provides a progress bar for training that displays
loss and other metrics in the terminal.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from tqdm import tqdm


class ConsoleProgress:
    """Console progress bar for training using tqdm.

    Displays a progress bar with current loss and average loss metrics.
    Can be used as an iterator over training data.

    Args:
        data: Iterable data source (e.g., DataLoader).
        name: Optional name prefix for the progress bar description.

    Example:
        >>> progress = ConsoleProgress(dataloader, name="train")
        >>> progress.start_epoch(0)
        >>> for batch in progress:
        ...     loss = train_step(batch)
        ...     progress.update({"loss": loss})
    """

    def __init__(
        self: ConsoleProgress,
        data: Iterable,
        name: str = "",
    ) -> None:
        self._data = data
        self._name = name
        self._pbar: tqdm | None = None
        self._current_epoch = 0
        self._current_loss = 0.0
        self._average_loss = 0.0

    def _format_description(self) -> str:
        """Format the progress bar description string."""
        prefix = f"{self._name} " if self._name else ""
        return (
            f"Epoch {self._current_epoch}; "
            f"{prefix}loss {self._current_loss:.2E}; "
            f"avg_loss {self._average_loss:.2E}"
        )

    def start_epoch(self, epoch: int) -> None:
        """Initialize the progress bar for a new epoch.

        Args:
            epoch: The epoch number to display.
        """
        self._current_epoch = epoch
        self._current_loss = 0.0
        self._average_loss = 0.0
        self._pbar = tqdm(self._data, desc=self._format_description())

    def update(self, metrics: dict[str, float]) -> None:
        """Update the progress bar with new metrics.

        Args:
            metrics: Dictionary containing at least 'loss' key.
        """
        if self._pbar is None:
            raise RuntimeError(
                "start_epoch() must be called before update()"
            )

        if "loss" in metrics:
            self._current_loss = metrics["loss"]
            # Update running average
            step = self._pbar.n
            self._average_loss = (
                step * self._average_loss + self._current_loss
            ) / (step + 1)

        self._pbar.set_description(self._format_description())

    def __iter__(self) -> Iterator:
        """Iterate over the data with progress display.

        Returns:
            Iterator over the data.

        Raises:
            RuntimeError: If start_epoch() was not called first.
        """
        if self._pbar is None:
            raise RuntimeError(
                "start_epoch() must be called before iteration"
            )
        return iter(self._pbar)

    def log_step(self, metrics: dict[str, float]) -> None:
        """TrainingLogger protocol: log step metrics.

        Args:
            metrics: Dictionary of metric names to values.
        """
        self.update(metrics)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """TrainingLogger protocol: log epoch metrics.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values.
        """
        # Console progress already shows metrics during iteration
        pass

    def close(self) -> None:
        """TrainingLogger protocol: clean up resources."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
