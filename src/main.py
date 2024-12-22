#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.backends.mps
from rich.console import Console

from config import load_config
from pipeline import run_pipeline

console = Console()

def verify_mps() -> None:
    """Verify MPS availability for Apple Silicon."""
    if not torch.backends.mps.is_available():
        console.print("[bold red]Error: This tool requires an Apple Silicon Mac with Metal Performance Shaders (MPS).[/bold red]")
        sys.exit(1)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 3 fine-tuning tool optimized for Apple Silicon"
    )
    
    parser.add_argument(
        "command",
        choices=["train", "export"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to training images directory"
    )
    
    parser.add_argument(
        "--prompt",
        help="Training prompt (e.g., 'a photo of sks dog')"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution for training"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory"
    )

    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.command == "train":
        if not args.data:
            raise ValueError("--data is required for training")
        if not args.prompt:
            raise ValueError("--prompt is required for training")
        if not args.data.exists():
            raise ValueError(f"Data directory does not exist: {args.data}")

def main(args: list = None) -> int:
    verify_mps()
    
    parser = create_parser()
    args = parser.parse_args(args)
    
    try:
        validate_args(args)
        
        # Load configuration
        config = load_config(args.config) if args.config else load_config()
        
        # Run pipeline
        run_pipeline(vars(args), config)
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())