import torch.backends.mps
from typing import Dict, Any

from .prepare import setup
from .train import run as run_training
from .optimize import run as run_optimization
from .export import save as save_model

def verify_mps() -> None:
    """Verify MPS availability."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "This tool requires an Apple Silicon Mac with Metal Performance Shaders (MPS) support."
        )

def run_pipeline(args: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """Run the complete fine-tuning pipeline."""
    verify_mps()
    
    if config is None:
        config = {}
    
    # Run pipeline stages
    config['components'] = setup(args, config)
    config['components'] = run_training(args, config)
    config['components'] = run_optimization(args, config)
    save_model(args, config)

__all__ = ['run_pipeline']