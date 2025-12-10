"""Training loop orchestration.

This module provides a high-level training loop that coordinates
data iteration, optimization, and logging.
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator, Sequence

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .tracker import LossTracker
from ..logging import ConsoleProgress, TrainingLogger


class TrainingLoop:
    """Orchestrates training with optional logging.

    Coordinates data iteration, loss tracking, optimization steps,
    and logging to multiple backends. Separates concerns that were
    previously mixed in ProgressBar.

    Args:
        data: Iterable data source (e.g., DataLoader or list).
        optimizer: Optional optimizer for gradient updates.
        scheduler: Optional learning rate scheduler.
        loggers: Optional list of TrainingLogger implementations.
        name: Optional name for this training loop (e.g., "train", "val").
        shuffle: Whether to shuffle data each epoch (only works for sequences).

    Example:
        >>> from dlu.training import TrainingLoop
        >>> from dlu.logging import WandbLogger
        >>>
        >>> loop = TrainingLoop(
        ...     dataloader,
        ...     optimizer=optimizer,
        ...     loggers=[WandbLogger("my-project")],
        ...     name="train",
        ... )
        >>> for epoch_num in range(num_epochs):
        ...     for batch in loop.epoch():
        ...         loss = model(batch)
        ...         loop.step(loss)
    """

    def __init__(
        self: TrainingLoop,
        data: Iterable,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        loggers: list[TrainingLogger] | None = None,
        name: str = "",
        shuffle: bool = False,
    ) -> None:
        self._data = data
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loggers = loggers or []
        self._name = name
        self._shuffle = shuffle

        self._tracker = LossTracker(name)
        self._console = ConsoleProgress(data, name)
        self._in_epoch = False

    @property
    def current_epoch(self) -> int:
        """Current epoch number (0-indexed)."""
        return self._tracker.current_epoch

    @property
    def current_step(self) -> int:
        """Current step within the epoch."""
        return self._tracker.current_step

    @property
    def current_loss(self) -> float:
        """Most recent loss value."""
        return self._tracker.current_loss

    @property
    def average_loss(self) -> float:
        """Running average loss for current epoch."""
        return self._tracker.average_loss

    def epoch(self) -> Iterator:
        """Start a new epoch and return an iterator over the data.

        Call this at the start of each epoch. It handles:
        - Logging previous epoch metrics (if not first epoch)
        - Resetting trackers
        - Shuffling data (if enabled and data is a sequence)
        - Setting up progress bar

        Returns:
            Iterator over the training data.

        Example:
            >>> for batch in loop.epoch():
            ...     loss = model(batch)
            ...     loop.step(loss)
        """
        # Log previous epoch metrics
        if self._tracker.current_epoch >= 0:
            metrics = self._tracker.epoch_metrics
            for logger in self._loggers:
                logger.log_epoch(self._tracker.current_epoch, metrics)

        # Start new epoch
        self._tracker.start_epoch()

        # Shuffle if requested and data supports it
        if self._shuffle and isinstance(self._data, Sequence):
            data_list = list(self._data)
            random.shuffle(data_list)
            self._console._data = data_list

        # Initialize progress bar
        self._console.start_epoch(self._tracker.current_epoch)
        self._in_epoch = True

        # Zero gradients at epoch start
        if self._optimizer is not None:
            self._optimizer.zero_grad()

        return iter(self._console)

    def step(self, loss: torch.Tensor) -> None:
        """Execute a training step with the given loss.

        Handles:
        - Recording loss in tracker
        - Backward pass (if optimizer provided)
        - Optimizer step (if optimizer provided)
        - Scheduler step (if scheduler provided)
        - Logging to all backends

        Args:
            loss: The loss tensor to backpropagate.

        Raises:
            RuntimeError: If called before epoch().
        """
        if not self._in_epoch:
            raise RuntimeError("epoch() must be called before step()")

        # Extract loss value
        loss_value = loss.detach().item()

        # Update tracker
        self._tracker.update(loss_value)

        # Update console progress
        self._console.update({"loss": loss_value})

        # Log to external loggers
        metrics = self._tracker.metrics
        for logger in self._loggers:
            logger.log_step(metrics)

        # Optimization step
        if self._optimizer is not None:
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

        # Scheduler step
        if self._scheduler is not None:
            self._scheduler.step()

    def close(self) -> None:
        """Clean up resources and close all loggers.

        Call this when training is complete to ensure all
        loggers are properly closed.
        """
        self._console.close()
        for logger in self._loggers:
            logger.close()
