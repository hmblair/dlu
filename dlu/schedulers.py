"""Learning rate schedulers for training.

This module provides custom learning rate schedulers that extend
PyTorch's scheduler functionality.
"""
from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler
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
