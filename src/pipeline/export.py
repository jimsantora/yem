import os
from pathlib import Path
import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from datetime import datetime

def save_lora_weights(transformer, output_dir: str):
    """Save LoRA weights optimized for Apple Silicon."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get LoRA state dict
    lora_state_dict = get_peft_model_state_dict(transformer)
    
    # Save in safetensors format for better memory usage
    weights_path = os.path.join(output_dir, "lora_weights.safetensors")
    save_file(lora_state_dict, weights_path)
    return weights_path

def generate_sample(pipeline, prompt: str, output_dir: str):
    """Generate a sample image using MPS."""
    # Ensure we're using float16 for better MPS performance
    pipeline.to(torch_dtype=torch.float16)
    
    with torch.inference_mode():
        image = pipeline(
            prompt,
            num_inference_steps=30,  # Reduced for Apple Silicon
            guidance_scale=7.5
        ).images[0]
    
    # Save sample
    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, "sample.png")
    image.save(sample_path)
    return sample_path

def generate_card(args, output_dir: str, sample_path: str = None):
    """Generate a simple model card."""
    card = f"""# Apple Silicon Optimized SD3 LoRA

## Model Details
- Base Model: {args.model}
- Training Prompt: {args.prompt}
- Resolution: {args.resolution}
- Created: {datetime.now().strftime('%Y-%m-%d')}

## Usage (Apple Silicon)
```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "{args.model}", 
    torch_dtype=torch.float16
).to("mps")

pipe.load_lora_weights("path/to/weights")
image = pipe("{args.prompt}").images[0]
```
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)

def save(args, config):
    """Main export function."""
    output_dir = Path(args.output_dir or "output")
    components = config.get('components', {})
    
    # Save weights
    weights_path = save_lora_weights(components['transformer'], output_dir)
    
    # Generate sample
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16
    )
    pipeline.load_lora_weights(weights_path)
    
    sample_path = generate_sample(pipeline, args.prompt, output_dir)
    
    # Create card
    generate_card(args, output_dir, sample_path)
    
    # Cleanup
    del pipeline
    torch.mps.empty_cache()