"""Evaluation metrics for Energy-based Models.

This module provides comprehensive evaluation metrics for generative models
including FID, IS, LPIPS, and custom EBM-specific metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
from pathlib import Path
import os


class EBMMetrics:
    """Comprehensive evaluation metrics for Energy-based Models."""
    
    def __init__(
        self,
        device: torch.device,
        num_samples: int = 10000,
        batch_size: int = 100
    ) -> None:
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        # Initialize metrics
        self.fid = FrechetInceptionDistance(feature=2048).to(device)
        self.is_metric = InceptionScore().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True
        ).to(device)
        
        # Storage for computed metrics
        self.metrics_history = {
            'fid': [],
            'is_mean': [],
            'is_std': [],
            'lpips_diversity': [],
            'energy_real': [],
            'energy_fake': [],
            'energy_gap': []
        }
    
    def compute_fid(self, real_images: Tensor, fake_images: Tensor) -> float:
        """Compute Fréchet Inception Distance.
        
        Args:
            real_images: Real images tensor
            fake_images: Generated images tensor
            
        Returns:
            FID score (lower is better)
        """
        self.fid.reset()
        
        # Process in batches
        for i in range(0, len(real_images), self.batch_size):
            real_batch = real_images[i:i+self.batch_size]
            fake_batch = fake_images[i:i+self.batch_size]
            
            # Convert to [0, 1] range for FID
            real_batch = (real_batch + 1) / 2
            fake_batch = (fake_batch + 1) / 2
            
            self.fid.update(real_batch, real=True)
            self.fid.update(fake_batch, real=False)
        
        return self.fid.compute().item()
    
    def compute_inception_score(self, images: Tensor) -> Tuple[float, float]:
        """Compute Inception Score.
        
        Args:
            images: Generated images tensor
            
        Returns:
            Tuple of (mean IS, std IS)
        """
        self.is_metric.reset()
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            
            # Convert to [0, 1] range
            batch = (batch + 1) / 2
            
            self.is_metric.update(batch)
        
        is_score = self.is_metric.compute()
        return is_score[0].item(), is_score[1].item()
    
    def compute_lpips_diversity(self, images: Tensor) -> float:
        """Compute LPIPS diversity within generated samples.
        
        Args:
            images: Generated images tensor
            
        Returns:
            Average LPIPS distance (higher is more diverse)
        """
        if len(images) < 2:
            return 0.0
        
        # Sample pairs for efficiency
        num_pairs = min(1000, len(images) * (len(images) - 1) // 2)
        distances = []
        
        for _ in range(num_pairs):
            # Random pair
            idx1, idx2 = np.random.choice(len(images), 2, replace=False)
            img1 = images[idx1:idx1+1]
            img2 = images[idx2:idx2+1]
            
            # Convert to [0, 1] range
            img1 = (img1 + 1) / 2
            img2 = (img2 + 1) / 2
            
            distance = self.lpips(img1, img2)
            distances.append(distance.item())
        
        return np.mean(distances)
    
    def compute_energy_statistics(
        self,
        model: nn.Module,
        real_images: Tensor,
        fake_images: Tensor
    ) -> Dict[str, float]:
        """Compute energy-based statistics.
        
        Args:
            model: EBM model
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            Dictionary of energy statistics
        """
        model.eval()
        with torch.no_grad():
            energy_real = model.energy(real_images).mean().item()
            energy_fake = model.energy(fake_images).mean().item()
            energy_gap = energy_fake - energy_real
        
        return {
            'energy_real': energy_real,
            'energy_fake': energy_fake,
            'energy_gap': energy_gap
        }
    
    def evaluate_model(
        self,
        model: nn.Module,
        real_images: Tensor,
        fake_images: Tensor,
        save_plots: bool = True,
        save_dir: str = './assets/evaluation'
    ) -> Dict[str, float]:
        """Comprehensive model evaluation.
        
        Args:
            model: EBM model
            real_images: Real images for comparison
            fake_images: Generated images
            save_plots: Whether to save evaluation plots
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of all computed metrics
        """
        print("Computing evaluation metrics...")
        
        # Ensure we have enough samples
        min_samples = min(len(real_images), len(fake_images), self.num_samples)
        real_subset = real_images[:min_samples]
        fake_subset = fake_images[:min_samples]
        
        # Compute metrics
        metrics = {}
        
        # FID
        print("Computing FID...")
        metrics['fid'] = self.compute_fid(real_subset, fake_subset)
        
        # Inception Score
        print("Computing Inception Score...")
        is_mean, is_std = self.compute_inception_score(fake_subset)
        metrics['is_mean'] = is_mean
        metrics['is_std'] = is_std
        
        # LPIPS Diversity
        print("Computing LPIPS diversity...")
        metrics['lpips_diversity'] = self.compute_lpips_diversity(fake_subset)
        
        # Energy statistics
        print("Computing energy statistics...")
        energy_stats = self.compute_energy_statistics(model, real_subset, fake_subset)
        metrics.update(energy_stats)
        
        # Update history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Save plots if requested
        if save_plots:
            self._save_evaluation_plots(real_subset, fake_subset, metrics, save_dir)
        
        return metrics
    
    def _save_evaluation_plots(
        self,
        real_images: Tensor,
        fake_images: Tensor,
        metrics: Dict[str, float],
        save_dir: str
    ) -> None:
        """Save evaluation plots and visualizations."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to [0, 1] range for visualization
        real_viz = (real_images + 1) / 2
        fake_viz = (fake_images + 1) / 2
        
        # Sample comparison grid
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('Real vs Generated Samples')
        
        for i in range(8):
            # Real images
            real_img = real_viz[i].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(real_img)
            axes[0, i].set_title(f'Real {i+1}')
            axes[0, i].axis('off')
            
            # Generated images
            fake_img = fake_viz[i].permute(1, 2, 0).cpu().numpy()
            axes[1, i].imshow(fake_img)
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'sample_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Metrics plot
        if len(self.metrics_history['fid']) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # FID
            axes[0, 0].plot(self.metrics_history['fid'])
            axes[0, 0].set_title('FID (lower is better)')
            axes[0, 0].set_xlabel('Evaluation Step')
            axes[0, 0].set_ylabel('FID')
            
            # Inception Score
            axes[0, 1].plot(self.metrics_history['is_mean'])
            axes[0, 1].set_title('Inception Score (higher is better)')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('IS')
            
            # Energy Gap
            axes[1, 0].plot(self.metrics_history['energy_gap'])
            axes[1, 0].set_title('Energy Gap (higher is better)')
            axes[1, 0].set_xlabel('Evaluation Step')
            axes[1, 0].set_ylabel('Energy Gap')
            
            # LPIPS Diversity
            axes[1, 1].plot(self.metrics_history['lpips_diversity'])
            axes[1, 1].set_title('LPIPS Diversity (higher is better)')
            axes[1, 1].set_xlabel('Evaluation Step')
            axes[1, 1].set_ylabel('LPIPS')
            
            plt.tight_layout()
            plt.savefig(save_path / 'metrics_history.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print formatted metrics."""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"FID: {metrics['fid']:.4f} (lower is better)")
        print(f"Inception Score: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f} (higher is better)")
        print(f"LPIPS Diversity: {metrics['lpips_diversity']:.4f} (higher is better)")
        print(f"Energy Real: {metrics['energy_real']:.4f}")
        print(f"Energy Fake: {metrics['energy_fake']:.4f}")
        print(f"Energy Gap: {metrics['energy_gap']:.4f} (higher is better)")
        print("="*50)


def create_metrics_table(metrics_history: Dict[str, List[float]]) -> str:
    """Create a formatted metrics table for comparison.
    
    Args:
        metrics_history: Dictionary of metric histories
        
    Returns:
        Formatted table string
    """
    if not metrics_history['fid']:
        return "No metrics available yet."
    
    # Get latest values
    latest = {k: v[-1] if v else 0.0 for k, v in metrics_history.items()}
    
    table = f"""
| Metric | Value | Best | Trend |
|--------|-------|------|-------|
| FID | {latest['fid']:.4f} | {min(metrics_history['fid']):.4f} | {'↗' if len(metrics_history['fid']) > 1 and latest['fid'] > metrics_history['fid'][-2] else '↘'} |
| IS | {latest['is_mean']:.4f} | {max(metrics_history['is_mean']):.4f} | {'↗' if len(metrics_history['is_mean']) > 1 and latest['is_mean'] > metrics_history['is_mean'][-2] else '↘'} |
| LPIPS | {latest['lpips_diversity']:.4f} | {max(metrics_history['lpips_diversity']):.4f} | {'↗' if len(metrics_history['lpips_diversity']) > 1 and latest['lpips_diversity'] > metrics_history['lpips_diversity'][-2] else '↘'} |
| Energy Gap | {latest['energy_gap']:.4f} | {max(metrics_history['energy_gap']):.4f} | {'↗' if len(metrics_history['energy_gap']) > 1 and latest['energy_gap'] > metrics_history['energy_gap'][-2] else '↘'} |
"""
    return table
