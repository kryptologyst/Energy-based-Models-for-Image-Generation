"""Sampling script for Energy-based Models.

This script provides utilities for generating samples from trained EBM models
with various sampling methods and parameters.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ebm import EnergyBasedModel
from configs.config import load_config, get_device_config, setup_reproducibility


def load_model(checkpoint_path: str, device: torch.device) -> EnergyBasedModel:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded EBM model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = EnergyBasedModel(
            in_channels=config.model.in_channels,
            base_channels=config.model.base_channels,
            num_layers=config.model.num_layers,
            image_size=config.model.image_size
        )
    else:
        # Default config if not available
        model = EnergyBasedModel()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_samples(
    model: EnergyBasedModel,
    num_samples: int = 64,
    method: str = 'langevin',
    steps: int = 100,
    step_size: float = 0.01,
    noise_scale: float = 0.01,
    device: torch.device = None
) -> torch.Tensor:
    """Generate samples using specified method.
    
    Args:
        model: Trained EBM model
        num_samples: Number of samples to generate
        method: Sampling method ('langevin' or 'mcmc')
        steps: Number of sampling steps
        step_size: Step size for sampling
        noise_scale: Noise scale for Langevin dynamics
        device: Device to run on
        
    Returns:
        Generated samples tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        if method == 'langevin':
            samples = model.langevin_sample(
                batch_size=num_samples,
                steps=steps,
                step_size=step_size,
                noise_scale=noise_scale,
                device=device
            )
        elif method == 'mcmc':
            samples = model.mcmc_sample(
                batch_size=num_samples,
                steps=steps,
                step_size=step_size,
                device=device
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


def save_samples_grid(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    title: Optional[str] = None
) -> None:
    """Save samples as image grid.
    
    Args:
        samples: Generated samples tensor
        save_path: Path to save image
        nrow: Number of images per row
        title: Optional title for the plot
    """
    # Convert from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(
        samples, nrow=nrow, padding=2, normalize=False
    )
    
    # Save image
    torchvision.utils.save_image(grid, save_path)
    
    # Also create matplotlib figure with title
    if title:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        # Save with title
        title_path = Path(save_path).with_suffix('.png')
        plt.savefig(title_path, dpi=150, bbox_inches='tight')
        plt.close()


def interpolate_samples(
    model: EnergyBasedModel,
    start_sample: torch.Tensor,
    end_sample: torch.Tensor,
    num_steps: int = 10,
    method: str = 'langevin',
    steps: int = 50,
    step_size: float = 0.01,
    device: torch.device = None
) -> torch.Tensor:
    """Interpolate between two samples.
    
    Args:
        model: Trained EBM model
        start_sample: Starting sample
        end_sample: Ending sample
        num_steps: Number of interpolation steps
        method: Sampling method for refinement
        steps: Number of sampling steps
        step_size: Step size for sampling
        device: Device to run on
        
    Returns:
        Interpolated samples
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Linear interpolation in latent space
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated = []
    
    for alpha in alphas:
        # Linear interpolation
        sample = (1 - alpha) * start_sample + alpha * end_sample
        
        # Refine using sampling
        if method == 'langevin':
            sample = model.langevin_sample(
                batch_size=1,
                steps=steps,
                step_size=step_size,
                device=device
            )
        elif method == 'mcmc':
            sample = model.mcmc_sample(
                batch_size=1,
                steps=steps,
                step_size=step_size,
                device=device
            )
        
        interpolated.append(sample)
    
    return torch.cat(interpolated, dim=0)


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate samples from EBM")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./assets/samples",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--num_samples", type=int, default=64,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--method", type=str, default="langevin",
        choices=["langevin", "mcmc"],
        help="Sampling method"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--step_size", type=float, default=0.01,
        help="Step size for sampling"
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.01,
        help="Noise scale for Langevin dynamics"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--interpolate", action="store_true",
        help="Generate interpolation between random samples"
    )
    parser.add_argument(
        "--interpolation_steps", type=int, default=10,
        help="Number of interpolation steps"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples using {args.method}...")
    samples = generate_samples(
        model=model,
        num_samples=args.num_samples,
        method=args.method,
        steps=args.steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        device=device
    )
    
    # Save samples
    samples_path = output_dir / f"samples_{args.method}_{args.steps}steps.png"
    save_samples_grid(
        samples,
        str(samples_path),
        nrow=8,
        title=f"Generated Samples ({args.method}, {args.steps} steps)"
    )
    print(f"Samples saved to: {samples_path}")
    
    # Generate interpolation if requested
    if args.interpolate:
        print("Generating interpolation...")
        
        # Generate two random samples for interpolation
        start_sample = generate_samples(
            model=model,
            num_samples=1,
            method=args.method,
            steps=args.steps,
            step_size=args.step_size,
            noise_scale=args.noise_scale,
            device=device
        )
        
        end_sample = generate_samples(
            model=model,
            num_samples=1,
            method=args.method,
            steps=args.steps,
            step_size=args.step_size,
            noise_scale=args.noise_scale,
            device=device
        )
        
        # Interpolate
        interpolated = interpolate_samples(
            model=model,
            start_sample=start_sample,
            end_sample=end_sample,
            num_steps=args.interpolation_steps,
            method=args.method,
            steps=args.steps,
            step_size=args.step_size,
            device=device
        )
        
        # Save interpolation
        interp_path = output_dir / f"interpolation_{args.method}_{args.steps}steps.png"
        save_samples_grid(
            interpolated,
            str(interp_path),
            nrow=args.interpolation_steps,
            title=f"Interpolation ({args.method}, {args.steps} steps)"
        )
        print(f"Interpolation saved to: {interp_path}")
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
