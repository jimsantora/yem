# Yem: SD3 Fine-tuning Tool for Apple Silicon

## Project Overview
Create a specialized Python application for fine-tuning Stable Diffusion 3 models on Apple Silicon using LoRA. The tool will be called "yem" and will leverage existing HuggingFace infrastructure while optimizing specifically for M-series chips.

## Core Architecture

### Base Components
The project will build upon the existing `train_dreambooth_lora_sd3.py` implementation, with Apple Silicon-specific optimizations. 

### Main Class Structure
```python
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
import torch.backends.mps

class Yem:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.accelerator = Accelerator(mixed_precision="fp16")
```

### Training Configuration
```python
def setup_training(self, 
                  model_path: str,
                  instance_data: str,
                  prompt: str,
                  rank: int = 4):
    """Configure LoRA training settings"""
    self.lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=[
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0"
        ],
        init_lora_weights="gaussian"
    )
```

### Memory Management
```python
def optimize_memory(self):
    """Apply M-series specific optimizations"""
    # Cache VAE outputs
    self.cache_latents = True
    
    # Enable gradient checkpointing 
    self.enable_grad_checkpointing = True
    
    # Monitor thermal state
    self.monitor_thermal = True
```

## Core Functionality Requirements

### 1. Training Flow
- Utilize FlowMatchEulerDiscreteScheduler from existing implementation
- Maintain DreamBooth dataset structure
- Keep validation pipeline intact

### 2. Apple Silicon Optimizations
- Auto-detect MPS availability
- Monitor thermal state
- Implement memory management for unified memory
- Auto-adjust batch sizes based on available memory

### 3. User Interface
- Simple CLI using argparse
- Progress bar using tqdm
- Memory monitoring display
- Thermal status indicator

### 4. Output Handling
- Save LoRA weights in safetensors format
- Generate validation images
- Create model cards for sharing
- Checkpoint management

## Implementation Notes

### 1. Code Reuse Strategy
- Leverage DreamBooth dataset class as-is
- Keep LoRA configuration structure
- Maintain training loop logic

### 2. Apple-Specific Features
- MPS device mapping
- Thermal monitoring
- Memory management
- Auto batch-size adjustment

### 3. Usability Focus
- Minimal required parameters 
- Sensible defaults for M-series chips
- Clear error messages
- Automatic recovery strategies

## Key Dependencies
```python
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline
)
from accelerate import Accelerator
from peft import LoraConfig
```

## Testing Strategy

### 1. Functional Testing
- Verify MPS support
- Monitor memory usage patterns
- Test thermal throttling handling
- Validate training stability

### 2. Output Validation
- LoRA weight format verification
- Training log completeness
- Validation image quality
- Model card generation

## Expected Outputs
1. Trained LoRA weights
2. Training logs
3. Validation images
4. Model card for sharing

## Usage Example
```bash
python -m yem train \
    --model "stabilityai/stable-diffusion-3" \
    --data "path/to/images" \
    --prompt "a photo of sks dog" \
    --resolution 1024 \
    --batch-size auto
```

## Notes for Implementation
The core goal is to create a focused tool that makes SD3 fine-tuning accessible on Apple Silicon while handling all the complex memory and thermal management behind the scenes. Build upon the robust foundation in `train_dreambooth_lora_sd3.py` while adding the necessary Apple Silicon optimizations and wrapping it in a user-friendly interface.