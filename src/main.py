#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.backends.mps
from rich.console import Console
from rich.progress import Progress

from config import load_config
from pipeline import prepare, train, optimize, export


console = Console()


def verify_apple_silicon() -> None:
    """Verify MPS availability for Apple Silicon."""
    if not torch.backends.mps.is_available():
        console.print("[bold red]Error: This tool requires an Apple Silicon Mac with Metal Performance Shaders (MPS).[/bold red]")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yem: Stable Diffusion 3 fine-tuning tool optimized for Apple Silicon"
    )
    
    parser.add_argument(
        "command",
        choices=["train", "export"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Base model to fine-tune (default: stabilityai/stable-diffusion-3-medium-diffusers)"
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
        help="Image resolution for training (default: 1024)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for training (default: auto)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
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


def determine_batch_size() -> int:
    """Auto-determine optimal batch size for M-series chips."""
    # Conservative defaults for M-series chips
    return 4  # Safe default for most M-series devices


def main(args: Optional[list] = None) -> int:
    # Verify Apple Silicon is available first
    verify_apple_silicon()
    
    parser = create_parser()
    args = parser.parse_args(args)
    
    try:
        validate_args(args)
        
        console.print("[bold green]Using Apple Silicon MPS[/bold green]")
        
        # Load configuration
        config = load_config(args.config) if args.config else {}
        
        # Auto-determine batch size if needed
        if args.batch_size == "auto":
            args.batch_size = determine_batch_size()
            console.print(f"[bold blue]Auto-selected batch size:[/bold blue] {args.batch_size}")
        else:
            args.batch_size = int(args.batch_size)
        
        if args.command == "train":
            # Pipeline stages
            with Progress() as progress:
                task = progress.add_task("Preparing...", total=4)
                
                prepare.setup(args, config)
                progress.update(task, advance=1, description="Training...")
                
                train.run(args, config)
                progress.update(task, advance=1, description="Optimizing...")
                
                optimize.run(args, config)
                progress.update(task, advance=1, description="Exporting...")
                
                export.save(args, config)
                progress.update(task, advance=1, description="Complete!")
                
        elif args.command == "export":
            export.save(args, config)
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())