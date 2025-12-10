"""Neural network building blocks.

This module provides common neural network components for deep learning:
- DenseNetwork: Configurable multi-layer perceptron
- Attention: Multi-head self-attention mechanism
- Transformer: Complete transformer encoder block
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


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


class Attention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads,
    using PyTorch's efficient `scaled_dot_product_attention`.

    Args:
        embed_dim: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.

    Example:
        >>> attn = Attention(embed_dim=256, num_heads=8)
        >>> x = torch.randn(100, 256)  # sequence of 100 tokens
        >>> output = attn(x)  # shape: (100, 256)
    """

    def __init__(
        self: Attention,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not embed_dim % num_heads == 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible "
                f"by the number of heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.input = DenseNetwork(embed_dim, 3 * embed_dim)
        self.output = DenseNetwork(num_heads * self.head_dim, embed_dim)

    def split(
        self: Attention,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Reshape tensor for multi-head attention.

        Args:
            x: Tensor of shape (seq_len, embed_dim).

        Returns:
            Tensor of shape (num_heads, seq_len, head_dim).
        """
        seq_len, _ = x.size()
        return x.view(
            seq_len,
            self.num_heads,
            self.head_dim,
        ).permute(1, 0, 2)

    def forward(
        self: Attention,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing self-attention.

        Args:
            x: Input tensor of shape (seq_len, embed_dim).
            bias: Optional attention bias/mask of shape
                (num_heads, seq_len, seq_len) or broadcastable.

        Returns:
            Output tensor of shape (seq_len, embed_dim).
        """
        n, _ = x.size()

        # Project to query, key, value
        query, key, value = self.input(x).chunk(3, dim=-1)
        query, key, value = [self.split(t) for t in (query, key, value)]

        # Compute attention
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=bias,
            dropout_p=self.dropout,
        )

        # Reshape and project output
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        return self.output(attn_output.view(n, self.embed_dim))


class Transformer(nn.Module):
    """Transformer encoder block.

    Combines token embedding, self-attention, and feed-forward network
    into a complete transformer encoder.

    Args:
        embeddings: Vocabulary size for the embedding layer.
        hidden_dim: Hidden dimension (embedding and attention dimension).
        out_dim: Output dimension.
        heads: Number of attention heads.
        dropout: Dropout probability.
        transition_factor: Factor to multiply hidden_dim for FFN intermediate size.

    Example:
        >>> transformer = Transformer(
        ...     embeddings=10000,
        ...     hidden_dim=256,
        ...     out_dim=10,
        ...     heads=8,
        ... )
        >>> tokens = torch.randint(0, 10000, (100,))  # sequence of 100 tokens
        >>> output = transformer(tokens)  # shape: (100, 10)
    """

    def __init__(
        self: Transformer,
        embeddings: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        transition_factor: int = 4,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(embeddings, hidden_dim)
        self.attn = Attention(hidden_dim, heads, dropout)
        self.out = DenseNetwork(
            hidden_dim, out_dim, [transition_factor * hidden_dim]
        )

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input token indices of shape (seq_len,).
            bias: Optional attention bias/mask.

        Returns:
            Output tensor of shape (seq_len, out_dim).
        """
        x = self.embedding(x)
        x = self.attn(x, bias)
        return self.out(x)
