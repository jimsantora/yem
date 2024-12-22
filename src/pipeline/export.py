import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from datetime import datetime

def save_lora_weights(
    transformer,
    text_encoder_one=None,
    text_encoder_two=None,
    output_dir: str = "output"
) -> str:
    """Save LoRA weights."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get LoRA state dict from transformer
    transformer_lora_layers = get_peft_model_state_dict(transformer)
    
    # Get LoRA state dict from text encoders if available
    text_encoder_lora_layers = None
    text_encoder_2_lora_layers = None
    
    if text_encoder_one is not None:
        text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one)
    if text_encoder_two is not None:
        text_encoder_2_lora_layers = get_peft_model_state_dict(text_encoder_two)
    
    # Save weights
    StableDiffusion3Pipeline.save_lora_weights(
        save_directory=output_dir,
        transformer_lora_layers=transformer_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    )
    
    return output_dir

def generate_sample(
    pipeline: StableDiffusion3Pipeline,
    prompt: str,
    output_dir: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5
) -> str:
    """Generate a sample image."""
    # Generate image
    with torch.inference_mode():
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    
    # Save image
    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, "sample.png")
    image.save(sample_path)
    
    return sample_path

def generate_model_card(
    args: Dict[str, Any],
    output_dir: str,
    sample_path: Optional[str] = None
) -> None:
    """Generate a model card."""
    card_content = f"""# LoRA Fine-tuned Stable Diffusion 3 Model

## Model Details
- Base Model: {args["model"]}
- Training Prompt: {args["prompt"]}
- Resolution: {args["resolution"]}
- Created: {datetime.now().strftime('%Y-%m-%d')}

## Usage with 🧨 diffusers
```python
from diffusers import StableDiffusion3Pipeline
import torch

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "{args["model"]}",
    torch_dtype=torch.float16
)
pipeline.load_lora_weights("{output_dir}")
image = pipeline("{args["prompt"]}").images[0]
```

## Training Details
- LoRA rank: 4
- Training resolution: {args["resolution"]}
- Batch size: {args["batch_size"]}
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card_content)

def save_pipeline(args: Dict[str, Any], components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save the trained model and generate samples."""
    output_dir = args.get("output_dir", "output")
    
    # Save LoRA weights
    save_lora_weights(
        components["transformer"],
        components.get("text_encoder_one"),
        components.get("text_encoder_two"),
        output_dir
    )
    
    # Load pipeline for sampling
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args["model"],
        torch_dtype=torch.float16
    )
    pipeline.load_lora_weights(output_dir)
    
    if torch.backends.mps.is_available():
        pipeline = pipeline.to("mps")
    
    # Generate sample
    sample_path = generate_sample(
        pipeline,
        args["prompt"],
        output_dir,
        guidance_scale=config["validation"]["guidance_scale"]
    )
    
    # Generate model card
    generate_model_card(args, output_dir, sample_path)
    
    # Clean up
    del pipeline
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()