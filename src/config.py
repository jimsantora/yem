from pathlib import Path
from typing import Any, Dict, Optional
import yaml

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults if not found."""
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
        "system": {
            "thermal_throttling": {
                "enabled": True,
                "max_temp": 90,  # Celsius
                "cooldown_time": 60  # seconds
            },
            "memory_monitoring": {
                "enabled": True,
                "warning_threshold": 0.85  # percentage
            }
        },
        "validation": {
            "enabled": True,
            "frequency": 20,  # epochs
            "num_images": 4,
            "guidance_scale": 7.5
        },
        "export": {
            "format": "safetensors",
            "half_precision": True,
            "include_metadata": True
        }
    }

    if config_path is None:
        return default_config

    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge user config with defaults
        merged_config = deep_merge(default_config, user_config)
        return merged_config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}, using defaults. Error: {e}")
        return default_config

def deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries, with update values taking precedence."""
    merged = base.copy()
    
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
            
    return merged
