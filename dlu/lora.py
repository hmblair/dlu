"""LoRA (Low-Rank Adaptation) for PyTorch models.

Provides drop-in LoRA injection for nn.Linear layers, controlled
by regex patterns over module names.

Reference: Hu et al. (2021) https://arxiv.org/abs/2106.09685

Example:
    >>> from dlu.lora import LoRAConfig, inject_lora, freeze_base
    >>> config = LoRAConfig(rank=8, targets=[r"\\.attn\\.linear_(q|k|v)$"])
    >>> injected = inject_lora(model, config)
    >>> freeze_base(model)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor. Effective scale is alpha / rank.
        targets: List of regex patterns matching module names to inject into.
        dropout: Dropout applied to the LoRA path.
    """

    rank: int = 8
    alpha: float = 16.0
    targets: list[str] = field(default_factory=list)
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    The output is: base(x) + (dropout(x) @ A^T @ B^T) * (alpha / rank)

    Args:
        base: The original nn.Linear to wrap.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor.
        dropout: Dropout probability on the LoRA path.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def _get_parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = model.get_submodule(parts[0])
        return parent, parts[1]
    return model, parts[0]


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Inject LoRA adapters into matching linear layers.

    Replaces each nn.Linear whose fully-qualified name matches any
    pattern in ``config.targets`` with a LoRALinear wrapper.

    Args:
        model: The model to modify in-place.
        config: LoRA configuration.

    Returns:
        List of module names that were injected.
    """
    if not config.targets:
        return []

    compiled = [re.compile(p) for p in config.targets]
    injected = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(p.search(name) for p in compiled):
            continue

        parent, child_name = _get_parent_and_child(model, name)
        lora_module = LoRALinear(
            base=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, child_name, lora_module)
        injected.append(name)

    return injected


def freeze_base(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters in the model."""
    for name, param in model.named_parameters():
        if ".lora_A" not in name and ".lora_B" not in name:
            param.requires_grad = False


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA parameters from the model state dict."""
    return {
        k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters into a model that already has LoRA injected."""
    model.load_state_dict(state_dict, strict=False)


def count_lora_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and LoRA parameters.

    Returns:
        Dict with keys "total", "trainable", "lora".
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ".lora_A" in n or ".lora_B" in n
    )
    return {"total": total, "trainable": trainable, "lora": lora}
