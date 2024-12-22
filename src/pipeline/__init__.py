from typing import Dict, Any

from .prepare import setup_pipeline
from .train import run_training
from .export import save_pipeline

def run_pipeline(args: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Run the complete fine-tuning pipeline."""
    # Initialize components
    components = setup_pipeline(args, config)
    
    # Run training
    if args["command"] == "train":
        components = run_training(args, components, config)
    
    # Save pipeline
    save_pipeline(args, components, config)