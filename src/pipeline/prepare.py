from typing import Dict, Any, Tuple
import torch
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
from peft import LoraConfig

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    subfolder: str = "text_encoder"
):
    """Import the correct text encoder class."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def setup_tokenizers(model_path: str) -> Tuple[CLIPTokenizer, CLIPTokenizer, T5TokenizerFast]:
    """Initialize tokenizers."""
    tokenizer_one = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        model_path,
        subfolder="tokenizer_3",
    )
    return tokenizer_one, tokenizer_two, tokenizer_three

def setup_text_encoders(model_path: str):
    """Initialize text encoders."""
    text_encoder_cls_one = import_model_class_from_model_name_or_path(model_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_path, 
        subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        model_path, 
        subfolder="text_encoder_3"
    )
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_path, 
        subfolder="text_encoder"
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_path, 
        subfolder="text_encoder_2"
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        model_path, 
        subfolder="text_encoder_3"
    )
    
    return text_encoder_one, text_encoder_two, text_encoder_three

def setup_models(model_path: str, config: Dict[str, Any]) -> Tuple[AutoencoderKL, SD3Transformer2DModel]:
    """Initialize models with LoRA configuration."""
    # Initialize VAE
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
    )
    
    # Initialize transformer
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["training"]["lora"]["rank"],
        lora_alpha=config["training"]["lora"]["alpha"],
        target_modules=config["training"]["lora"]["target_modules"],
        init_lora_weights="gaussian",
    )
    
    # Add LoRA adapter to transformer
    transformer.add_adapter(lora_config)
    
    # Enable gradient checkpointing if configured
    if config["memory"]["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    
    return vae, transformer

def setup_pipeline(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all pipeline components."""
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        mixed_precision=config["training"]["mixed_precision"]
    )
    
    # Initialize components
    tokenizers = setup_tokenizers(args["model"])
    text_encoders = setup_text_encoders(args["model"])
    vae, transformer = setup_models(args["model"], config)
    
    # Setup scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args["model"],
        subfolder="scheduler"
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config["training"]["learning_rate"]
    )
    
    # Move models to device if using MPS
    if torch.backends.mps.is_available():
        vae.to("mps")
        transformer.to("mps")
        for encoder in text_encoders:
            encoder.to("mps")
    
    return {
        "accelerator": accelerator,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "vae": vae,
        "transformer": transformer,
        "scheduler": scheduler,
        "optimizer": optimizer,
    }