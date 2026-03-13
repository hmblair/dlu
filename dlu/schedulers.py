"""Learning rate schedulers for training.

This module provides custom learning rate schedulers that extend
PyTorch's scheduler functionality.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer


class LinearWarmupSqrtDecay(_LRScheduler):
    """Learning rate scheduler with linear warmup and inverse square-root decay.

    The learning rate follows two phases:
    1. Linear warmup: LR increases linearly from 0 to initial_lr over warmup_steps
    2. Decay: LR decays proportionally to 1/sqrt(step) after warmup

    This schedule is commonly used for transformer training (Vaswani et al., 2017).

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps for linear warmup phase.
        *args: Additional arguments passed to _LRScheduler.
        **kwargs: Additional keyword arguments passed to _LRScheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = LinearWarmupSqrtDecay(optimizer, warmup_steps=1000)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss = train_step(batch)
        ...         scheduler.step()
    """

    def __init__(
        self: LinearWarmupSqrtDecay,
        optimizer: Optimizer,
        warmup_steps: int,
        *args,
        **kwargs,
    ) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, *args, **kwargs)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current step.

        Returns:
            List of learning rates for each parameter group.
        """
        if self.last_epoch == -1:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_steps:
            return self._warmup_step()
        else:
            return self._decay_step()

    def _warmup_step(self) -> list[float]:
        """Compute learning rate during linear warmup phase.

        Returns:
            List of learning rates scaled linearly based on current step.
        """
        scale = (self.last_epoch + 1) / self.warmup_steps
        return [
            group['initial_lr'] * scale
            for group in self.optimizer.param_groups
        ]

    def _decay_step(self) -> list[float]:
        """Compute learning rate during inverse square-root decay phase.

        Returns:
            List of learning rates decayed by inverse square root of step.
        """
        scale = (self.last_epoch + 2) / (self.last_epoch + 1)
        return [
            group['lr'] * (scale ** (-1/2))
            for group in self.optimizer.param_groups
        ]


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay.

    Two-phase schedule:
    1. Linear warmup from 0 to initial_lr over num_warmup_steps
    2. Cosine decay from initial_lr to min_lr_ratio * initial_lr

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of initial LR.

    Returns:
        LambdaLR scheduler.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=10000
        ... )
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
