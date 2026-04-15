"""Neural network building blocks.

This module provides common neural network components for deep learning:
- DenseNetwork: Configurable multi-layer perceptron
- RadialBasisFunctions: Learnable Gaussian basis functions for scalar featurization
- RMSNorm: Root Mean Square Layer Normalization
- RotaryPositionEmbedding: Rotary Position Embeddings (RoPE)
- SwiGLU: Gated feedforward network with Swish activation
- MultiHeadAttention: Multi-head attention with optional RoPE
- TransformerBlock: Pre-LN transformer block
- Transformer: Full transformer encoder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetwork(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    A fully-connected neural network with customizable hidden layers,
    activation functions, and dropout.

    Args:
        in_size: Input feature dimension.
        out_size: Output feature dimension.
        hidden_sizes: List of hidden layer dimensions. Empty list means
            a single linear layer from in_size to out_size.
        bias: Whether to use bias in linear layers.
        dropout: Dropout probability applied after each hidden layer.
        activation: Activation function applied after each hidden layer.

    Example:
        >>> mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
        >>> x = torch.randn(32, 64)  # batch of 32
        >>> output = mlp(x)  # shape: (32, 10)
    """

    def __init__(
        self: DenseNetwork,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int] | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []
        if activation is None:
            activation = nn.ReLU()

        features = [in_size] + hidden_sizes + [out_size]

        layers = []
        for l1, l2 in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(l1, l2, bias))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (..., in_size).

        Returns:
            Output tensor of shape (..., out_size).
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)


class RadialBasisFunctions(nn.Module):
    """Radial basis functions with learnable centers and widths.

    Each function is a Gaussian parameterized by a learnable mean (mu) and
    inverse width (sigma). Useful for featurizing scalar inputs (e.g.
    distances) into a higher-dimensional space.

    Args:
        num_functions: Number of radial basis functions.

    Example:
        >>> rbf = RadialBasisFunctions(16)
        >>> distances = torch.randn(32, 100)  # (batch, num_distances)
        >>> features = rbf(distances)  # shape: (32, 100, 16)
    """

    def __init__(self, num_functions: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.randn(num_functions))
        self.sigma = nn.Parameter(torch.randn(num_functions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions at each input value.

        Args:
            x: Input tensor of shape (...).

        Returns:
            Tensor of shape (..., num_functions).
        """
        exp = (x[..., None] - self.mu) * self.sigma**2
        return torch.exp(-(exp**2)) * self.sigma.abs()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — normalizes by RMS without centering.

    Reference: Zhang & Sennrich (2019) https://arxiv.org/abs/1910.07467

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Encodes position by rotating query/key vectors, enabling relative
    position learning and length generalization.

    Reference: Su et al. (2021) https://arxiv.org/abs/2104.09864

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length for cached embeddings.
        base: Base for frequency computation.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq, head_dim).
            k: Key tensor of shape (batch, heads, seq, head_dim).
            seq_len: Sequence length (defaults to q.shape[2]).

        Returns:
            Rotated (q, k) tensors.
        """
        if seq_len is None:
            seq_len = q.shape[2]

        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """SwiGLU feedforward network.

    Gated Linear Unit with Swish activation. Generally outperforms
    GELU/ReLU FFN. Uses fused w1/w2 projection for GPU efficiency.

    Reference: Shazeer (2020) https://arxiv.org/abs/2002.05202

    Args:
        d_model: Input/output dimension.
        d_ff: Intermediate dimension (default: 4 * d_model, rounded to multiple of 64).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.split(self.d_ff, dim=-1)
        return self.dropout(self.w3(F.silu(x1) * x2))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and QK-Norm.

    Uses Flash Attention (scaled_dot_product_attention) when available.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        use_rope: Whether to use Rotary Position Embeddings.
        qk_norm: Whether to apply RMSNorm to queries and keys.
        is_causal: Whether to apply causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq, d_model).
            mask: Padding mask of shape (batch, seq) where True = masked.
            attn_bias: Attention bias of shape (batch, heads, seq, seq)
                or (batch, seq, seq). Added to attention scores.

        Returns:
            Output tensor of shape (batch, seq, d_model).
        """
        B, L, D = x.shape

        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q, k = self.rope(q, k, seq_len=L)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        combined_mask = None

        if mask is not None:
            combined_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1)
            combined_mask = combined_mask.float().masked_fill(
                combined_mask, float("-inf")
            )

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            if combined_mask is not None:
                combined_mask = combined_mask + attn_bias
            else:
                combined_mask = attn_bias

        if self.is_causal and combined_mask is not None:
            causal = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            causal_mask = causal.float().masked_fill(causal, float("-inf"))
            combined_mask = combined_mask + causal_mask

        if self.is_causal and combined_mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=combined_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with RMSNorm, optional RoPE, and SwiGLU.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        use_rope: Whether to use Rotary Position Embeddings.
        qk_norm: Whether to apply QK-Norm.
        is_causal: Whether to apply causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, num_heads, dropout, max_seq_len, use_rope, qk_norm, is_causal
        )
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask, attn_bias=attn_bias))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module):
    """Modern Transformer encoder.

    Pre-LN architecture with RMSNorm, optional RoPE, and SwiGLU,
    following best practices from LLaMA, GPT-NeoX, and PaLM.

    Args:
        d_model: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        use_rope: Whether to use Rotary Position Embeddings.
        qk_norm: Whether to apply QK-Norm for training stability.
        is_causal: Whether to apply causal masking.

    Example:
        >>> transformer = Transformer(d_model=256, num_layers=4, num_heads=8)
        >>> x = torch.randn(2, 100, 256)  # (batch, seq, d_model)
        >>> output = transformer(x)  # shape: (2, 100, 256)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    max_seq_len,
                    use_rope,
                    qk_norm,
                    is_causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process input through transformer layers.

        Args:
            x: Input tensor (batch, seq, d_model).
            mask: Padding mask (batch, seq) where True = masked.
            attn_bias: Attention bias (batch, heads, seq, seq).

        Returns:
            Output tensor (batch, seq, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask=mask, attn_bias=attn_bias)

        return self.final_norm(x)
