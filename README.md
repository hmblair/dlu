# DLU - Deep Learning Utilities

A collection of PyTorch utilities for deep learning research.

## Installation

```bash
pip install dlu-torch

# With optional wandb support
pip install "dlu-torch[wandb]"

# All extras
pip install "dlu-torch[all]"
```

## Quick Start

### Neural Network Modules

```python
from dlu import DenseNetwork, RadialBasisFunctions, Transformer

# Multi-layer perceptron
mlp = DenseNetwork(in_size=64, out_size=10, hidden_sizes=[128, 64])

# Learnable Gaussian basis functions (e.g. for distance featurization)
rbf = RadialBasisFunctions(num_functions=16)

# Transformer encoder (pre-LN, RMSNorm, RoPE, SwiGLU)
transformer = Transformer(d_model=256, num_layers=4, num_heads=8)
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

### LoRA

```python
from dlu.lora import LoRAConfig, inject_lora, freeze_base

config = LoRAConfig(rank=8, targets=[r"\.attn\.qkv_proj$"])
inject_lora(model, config)
freeze_base(model)
```

### Learning Rate Schedulers

```python
from dlu import LinearWarmupSqrtDecay, get_cosine_schedule_with_warmup

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
├── modules.py      - Neural network modules (DenseNetwork, RadialBasisFunctions,
│                     RMSNorm, RoPE, SwiGLU, MultiHeadAttention, Transformer)
├── lora.py         - LoRA injection for nn.Linear layers
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

## License

MIT
