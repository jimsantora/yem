import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from transformers import CLIPTokenizer, T5TokenizerFast
from peft import LoraConfig

def setup_tokenizers(model_path: str):
    """Initialize tokenizers for SD3."""
    return (
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer"),
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2"),
        T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")
    )

def setup_models(model_path: str):
    """Initialize models optimized for MPS."""
    # VAE stays in float32 for stability
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae"
    ).to("mps")
    
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer"
    ).to("mps")
    
    # Configure LoRA for efficient training
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=[
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0"
        ],
        init_lora_weights="gaussian"
    )
    
    transformer.add_adapter(lora_config)
    transformer.enable_gradient_checkpointing()
    
    return vae, transformer

def setup(args, config):
    """Initialize training components."""
    # Use no mixed precision for MPS
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision="no"
    )
    
    tokenizers = setup_tokenizers(args.model)
    vae, transformer = setup_models(args.model)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model, 
        subfolder="scheduler"
    )
    
    return {
        'accelerator': accelerator,
        'tokenizers': tokenizers,
        'vae': vae,
        'transformer': transformer,
        'scheduler': scheduler
    }