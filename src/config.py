from pathlib import Path
from typing import Any, Dict, Optional
import yaml

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with defaults."""
    default_config = {
        "training": {
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "warmup_steps": 500,
            "gradient_accumulation": 1,
            "mixed_precision": "fp16",
            "lora": {
                "rank": 4,
                "alpha": 4,
                "target_modules": [
                    "attn.to_q",
                    "attn.to_k", 
                    "attn.to_v",
                    "attn.to_out.0"
                ]
            }
        },
        "memory": {
            "cache_latents": True,
            "gradient_checkpointing": True,
            "attention_slicing": "auto",
            "vae_slicing": True
        },
        "validation": {
            "enabled": True,
            "frequency": 20,
            "num_images": 4,
            "guidance_scale": 7.5
        }
    }
    
    if config_path is None:
        return default_config
        
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        merged_config = deep_merge(default_config, user_config)
        return merged_config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}, using defaults. Error: {e}")
        return default_config

def deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
            
    return merged