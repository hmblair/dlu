# DLU - Deep Learning Utilities

A collection of PyTorch utilities for deep learning research.

## Installation

```bash
pip install -e .

# With optional wandb support
pip install -e ".[wandb]"
```

## Quick Start

### Neural Network Modules

```python
from dlu import DenseNetwork, Attention, Transformer

# Multi-layer perceptron
mlp = DenseNetwork(in_size=64, out_size=10, hidden_sizes=[128, 64])

# Multi-head attention
attn = Attention(embed_dim=256, num_heads=8)

# Transformer block
transformer = Transformer(
    embeddings=10000,
    hidden_dim=256,
    out_dim=10,
    heads=8,
)
```

### Training Loop

```python
from dlu.training import TrainingLoop
from dlu.logging import WandbLogger  # Optional

# Basic training loop
loop = TrainingLoop(dataloader, optimizer=optimizer, name="train")
for epoch in range(num_epochs):
    for batch in loop.epoch():
        loss = model(batch)
        loop.step(loss)

# With wandb logging
loggers = [WandbLogger("my-project")]
loop = TrainingLoop(
    dataloader,
    optimizer=optimizer,
    loggers=loggers,
    name="train",
)
```

### Learning Rate Schedulers

```python
from dlu import LinearWarmupSqrtDecay

scheduler = LinearWarmupSqrtDecay(optimizer, warmup_steps=1000)
```

### Utilities

```python
from dlu import params, normalize, plot_tensor

# Count parameters
num_params = params(model)
print(f"Model has {num_params:,} parameters")

# Normalize tensor (z-score or min-max)
normalized = normalize(tensor)
normalized_minmax = normalize(tensor, use_minmax=True)

# Plot tensor
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_tensor(ax, tensor)
```

## Package Structure

```
dlu/
├── modules.py      - Neural network modules (DenseNetwork, Attention, Transformer)
├── schedulers.py   - Learning rate schedulers
├── utils.py        - Utility functions (params)
├── transforms.py   - Data transformations (normalize)
├── plotting.py     - Plotting utilities (plot_tensor)
├── training/       - Training loop utilities
│   ├── tracker.py  - Loss tracking
│   └── loop.py     - Training loop orchestration
└── logging/        - Logging backends
    ├── base.py     - Logger protocol
    ├── console.py  - Console progress (tqdm)
    └── wandb.py    - Weights & Biases (optional)
```

## Related Projects

For geometric deep learning (SO(3)-equivariant networks, point cloud alignment), see [lr_geom](https://github.com/hmblair/lr_geom).

## License

CC BY-NC 4.0
