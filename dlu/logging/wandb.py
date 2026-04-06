"""Optional Weights & Biases integration.

This module provides wandb logging with lazy imports, so wandb is only
required when actually used. Users without wandb installed can still
use other parts of the library.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb

# Lazy import state
_wandb_module: Any = None
_wandb_available: bool | None = None


def _get_wandb() -> Any:
    """Lazily import wandb and cache the result.

    Returns:
        The wandb module if available, None otherwise.
    """
    global _wandb_module, _wandb_available

    if _wandb_available is None:
        try:
            import wandb as wb
            _wandb_module = wb
            _wandb_available = True
        except ImportError:
            _wandb_available = False

    return _wandb_module


def is_wandb_available() -> bool:
    """Check if wandb is installed and available.

    Returns:
        True if wandb can be imported, False otherwise.
    """
    _get_wandb()
    return _wandb_available or False


class WandbLogger:
    """Weights & Biases logger for training.

    Logs training metrics to wandb for visualization and tracking.
    Requires wandb to be installed (`pip install wandb`).

    Args:
        project: Name of the wandb project.
        **init_kwargs: Additional arguments passed to wandb.init().

    Raises:
        ImportError: If wandb is not installed.

    Example:
        >>> logger = WandbLogger("my-project", name="experiment-1")
        >>> logger.log_step({"loss": 0.5, "accuracy": 0.85})
        >>> logger.log_epoch(0, {"val_loss": 0.4})
        >>> logger.close()
    """

    def __init__(
        self: WandbLogger,
        project: str,
        **init_kwargs: Any,
    ) -> None:
        wb = _get_wandb()
        if wb is None:
            raise ImportError(
                "wandb is required for WandbLogger. "
                "Install with: pip install wandb"
            )

        self._wandb = wb
        self._run = wb.init(project=project, **init_kwargs)

    def log_step(self, metrics: dict[str, float]) -> None:
        """Log metrics for a single training step.

        Args:
            metrics: Dictionary of metric names to values.
        """
        self._run.log(metrics)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics at the end of an epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values.
        """
        self._run.log(metrics, step=epoch)

    def close(self) -> None:
        """Finish the wandb run and clean up resources."""
        if self._run is not None:
            self._run.finish()
            self._run = None
