"""Main training script for Energy-based Models.

This script provides a complete training pipeline with evaluation,
logging, and checkpointing for EBM experiments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ebm import EnergyBasedModel, EBMTrainer
from data.datasets import get_data_loaders
from utils.metrics import EBMMetrics
from configs.config import (
    load_config, save_config, get_experiment_config,
    get_device_config, setup_reproducibility
)


def setup_directories(config) -> None:
    """Create necessary directories."""
    paths = [
        config.paths.checkpoints,
        config.paths.samples,
        config.paths.logs,
        config.paths.configs
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_samples(
    samples: torch.Tensor,
    epoch: int,
    save_dir: str,
    prefix: str = "samples"
) -> None:
    """Save generated samples as image grid."""
    # Convert from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(
        samples, nrow=8, padding=2, normalize=False
    )
    
    # Save image
    save_path = Path(save_dir) / f"{prefix}_epoch_{epoch:04d}.png"
    torchvision.utils.save_image(grid, save_path)


def train_epoch(
    trainer: EBMTrainer,
    dataloader: DataLoader,
    epoch: int,
    config
) -> Dict[str, float]:
    """Train for one epoch."""
    trainer.model.train()
    
    epoch_losses = {'total': [], 'real': [], 'fake': []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (real_data, _) in enumerate(pbar):
        real_data = real_data.to(trainer.device)
        
        # Training step
        losses = trainer.train_step(real_data)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{losses['total']:.4f}",
            'Real': f"{losses['real']:.4f}",
            'Fake': f"{losses['fake']:.4f}"
        })
        
        # Store losses
        for key, value in losses.items():
            epoch_losses[key].append(value)
    
    # Return average losses
    return {key: np.mean(values) for key, values in epoch_losses.items()}


def evaluate_model(
    trainer: EBMTrainer,
    dataloader: DataLoader,
    metrics: EBMMetrics,
    config
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    trainer.model.eval()
    
    # Collect real images
    real_images = []
    with torch.no_grad():
        for batch_idx, (real_data, _) in enumerate(dataloader):
            real_images.append(real_data.to(trainer.device))
            if len(real_images) * dataloader.batch_size >= config.evaluation.num_samples:
                break
    
    real_images = torch.cat(real_images, dim=0)[:config.evaluation.num_samples]
    
    # Generate fake images
    fake_images = trainer.generate_samples(
        num_samples=config.evaluation.num_samples,
        steps=config.sampling.steps,
        step_size=config.sampling.step_size
    )
    
    # Compute metrics
    eval_metrics = metrics.evaluate_model(
        trainer.model,
        real_images,
        fake_images,
        save_plots=True,
        save_dir=config.paths.samples
    )
    
    return eval_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Energy-based Model")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Predefined experiment name"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate, don't train"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.experiment:
        config = get_experiment_config(args.experiment)
    else:
        config = load_config(args.config)
    
    # Setup reproducibility
    setup_reproducibility(config)
    
    # Setup device
    device = torch.device(get_device_config(config))
    print(f"Using device: {device}")
    
    # Setup directories
    setup_directories(config)
    
    # Save configuration
    config_path = Path(config.paths.configs) / "config.yaml"
    save_config(config, config_path)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        image_size=(config.model.image_size, config.model.image_size),
        val_split=config.data.val_split
    )
    
    print(f"Dataset: {config.data.dataset}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = EnergyBasedModel(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        image_size=config.model.image_size
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = EBMTrainer(
        model=model,
        device=device,
        lr=config.training.learning_rate,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        weight_decay=config.training.weight_decay
    )
    
    # Create metrics
    metrics = EBMMetrics(
        device=device,
        num_samples=config.evaluation.num_samples,
        batch_size=config.evaluation.batch_size
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    if args.eval_only:
        print("Evaluation only mode")
        eval_metrics = evaluate_model(trainer, val_loader, metrics, config)
        metrics.print_metrics(eval_metrics)
        return
    
    # Training loop
    print("Starting training...")
    best_fid = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # Train
        epoch_losses = train_epoch(trainer, train_loader, epoch, config)
        
        print(f"Epoch {epoch} - Loss: {epoch_losses['total']:.4f}")
        
        # Evaluation
        if (epoch + 1) % config.training.eval_every == 0:
            print("Evaluating...")
            eval_metrics = evaluate_model(trainer, val_loader, metrics, config)
            metrics.print_metrics(eval_metrics)
            
            # Save best model
            if eval_metrics['fid'] < best_fid:
                best_fid = eval_metrics['fid']
                checkpoint_path = Path(config.paths.checkpoints) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': epoch_losses['total'],
                    'fid': eval_metrics['fid'],
                    'config': config
                }, checkpoint_path)
                print(f"Saved best model with FID: {best_fid:.4f}")
        
        # Generate samples
        if (epoch + 1) % config.training.sample_every == 0:
            print("Generating samples...")
            samples = trainer.generate_samples(
                num_samples=config.sampling.num_samples,
                steps=config.sampling.steps,
                step_size=config.sampling.step_size
            )
            save_samples(
                samples, epoch, config.paths.samples, "training_samples"
            )
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = Path(config.paths.checkpoints) / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': epoch_losses['total'],
                'config': config
            }, checkpoint_path)
    
    print("Training completed!")
    
    # Final evaluation
    print("Final evaluation...")
    final_metrics = evaluate_model(trainer, test_loader, metrics, config)
    metrics.print_metrics(final_metrics)
    
    # Save final model
    final_checkpoint_path = Path(config.paths.checkpoints) / "final_model.pth"
    torch.save({
        'epoch': config.training.num_epochs - 1,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': epoch_losses['total'],
        'final_metrics': final_metrics,
        'config': config
    }, final_checkpoint_path)
    
    print(f"Final model saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
