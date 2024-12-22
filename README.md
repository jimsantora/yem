# Yem: Stable Diffusion 3 Fine-tuning for Apple Silicon 🍏🤖

Train Stable Diffusion 3 models efficiently on Apple Silicon using LoRA. Yem automatically handles memory management, thermal throttling, and performance optimizations specific to M-series chips. 

## Features

- 🚀 Optimized for Apple Silicon (M1/M2/M3)
- 🧠 Smart memory management and thermal monitoring
- ⚡️ Automatic batch size adjustment
- 🎯 Efficient LoRA fine-tuning
- 📦 Simple CLI interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yem
cd yem
```

2. Install in development mode:
```bash
pip install -e .
```

## Quick Start 🚀

Remember: All commands need either `train` or `export` as the first argument!

Train a model:
```bash
python main.py train \
    --data /path/to/images \
    --prompt "a photo of sks dog" \
    --resolution 1024
```

Export a model:
```bash
python main.py export \
    --model your/model/path
```

## CLI Options 🛠️

```
Commands:
train                   Run the full training pipeline
export                  Export a trained model

Options:
--model                 Base model (default: stabilityai/stable-diffusion-3-medium-diffusers)
--data                  Training images directory
--prompt               Training prompt
--resolution           Image resolution (default: 1024)
--batch-size           Batch size (default: auto)
--config               Path to config file
```

## Apple Silicon Optimizations 🍎

### Memory Management 🧠
- Uses unified memory architecture efficiently
- Automatic VAE caching
- Dynamic batch size adjustment based on available memory
- Strategic MPS cache clearing

### Performance Tuning ⚡️
- Mixed precision training (float16)
- Gradient checkpointing for larger batches
- Optimized transformer configurations
- Thermal throttling detection and management

### Hardware-Specific Features 🔧
- MPS backend utilization
- Thermal state monitoring
- Memory pressure detection
- Automatic recovery from thermal throttling

## Model Architecture 🏗️

Yem uses LoRA (Low-Rank Adaptation) to efficiently fine-tune SD3 models with these default attention configurations:

```python
target_modules = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0"
]
```

## Requirements 📋

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- torch >= 2.0.0
- diffusers >= 0.25.0

## License 📜

MIT