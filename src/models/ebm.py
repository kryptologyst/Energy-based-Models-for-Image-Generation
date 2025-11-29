"""Energy-based Model implementation for image generation.

This module implements a modern Energy-based Model (EBM) with proper energy function
learning, Langevin dynamics sampling, and MCMC-based generation.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization for stable training."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
                )
            )
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)


class EnergyBasedModel(nn.Module):
    """Modern Energy-based Model with residual architecture and spectral normalization.
    
    The model learns an energy function E(x) that assigns lower energy to real data
    and higher energy to generated/fake data. This enables both generation and
    discrimination tasks.
    
    Args:
        in_channels: Number of input channels (3 for RGB images)
        base_channels: Base number of channels for the network
        num_layers: Number of residual blocks
        image_size: Input image size (assumed square)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        image_size: int = 32
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.image_size = image_size
        
        # Initial convolution
        self.initial_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        channels = base_channels
        
        for i in range(num_layers):
            stride = 2 if i == num_layers // 2 else 1
            self.res_blocks.append(ResidualBlock(channels, channels, stride))
        
        # Global average pooling and final layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(channels, channels // 2))
        self.fc2 = nn.Linear(channels // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to compute energy scores.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Energy scores of shape (batch_size, 1)
        """
        # Initial convolution
        out = F.relu(self.initial_conv(x))
        
        # Residual blocks
        for block in self.res_blocks:
            out = block(out)
        
        # Global pooling and final layers
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        energy = self.fc2(out)
        
        return energy
    
    def energy(self, x: Tensor) -> Tensor:
        """Compute energy for input samples.
        
        Args:
            x: Input tensor
            
        Returns:
            Energy values (lower is better for real data)
        """
        return self.forward(x)
    
    def langevin_sample(
        self,
        batch_size: int,
        steps: int = 100,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate samples using Langevin dynamics.
        
        Args:
            batch_size: Number of samples to generate
            steps: Number of Langevin steps
            step_size: Step size for gradient updates
            noise_scale: Scale of noise added at each step
            device: Device to run on
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Initialize with random noise
        x = torch.randn(
            batch_size, self.in_channels, self.image_size, self.image_size,
            device=device, requires_grad=True
        )
        
        # Langevin dynamics
        for _ in range(steps):
            # Compute energy and gradients
            energy = self.energy(x).sum()
            grad = torch.autograd.grad(energy, x, create_graph=True)[0]
            
            # Update samples
            x = x - step_size * grad + noise_scale * torch.randn_like(x)
            
            # Clamp to valid range
            x = torch.clamp(x, -1, 1)
        
        return x.detach()
    
    def mcmc_sample(
        self,
        batch_size: int,
        steps: int = 100,
        step_size: float = 0.01,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate samples using MCMC with Metropolis-Hastings.
        
        Args:
            batch_size: Number of samples to generate
            steps: Number of MCMC steps
            step_size: Proposal step size
            device: Device to run on
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Initialize with random noise
        x = torch.randn(
            batch_size, self.in_channels, self.image_size, self.image_size,
            device=device
        )
        
        # MCMC sampling
        for _ in range(steps):
            # Propose new samples
            x_proposed = x + step_size * torch.randn_like(x)
            x_proposed = torch.clamp(x_proposed, -1, 1)
            
            # Compute energy difference
            with torch.no_grad():
                energy_current = self.energy(x).sum()
                energy_proposed = self.energy(x_proposed).sum()
                
                # Metropolis acceptance criterion
                log_alpha = -(energy_proposed - energy_current)
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                
                # Accept/reject
                accept = torch.rand(batch_size, device=device) < alpha
                x = torch.where(accept.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                               x_proposed, x)
        
        return x


class EBMTrainer:
    """Trainer for Energy-based Models with proper loss functions and sampling."""
    
    def __init__(
        self,
        model: EnergyBasedModel,
        device: torch.device,
        lr: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 1e-5
    ) -> None:
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        # Training statistics
        self.step = 0
        self.epoch = 0
        self.losses = {'total': [], 'real': [], 'fake': []}
    
    def compute_loss(self, real_data: Tensor, fake_data: Tensor) -> Dict[str, Tensor]:
        """Compute EBM loss using contrastive divergence.
        
        Args:
            real_data: Real data samples
            fake_data: Generated/fake data samples
            
        Returns:
            Dictionary containing loss components
        """
        # Compute energies
        energy_real = self.model.energy(real_data)
        energy_fake = self.model.energy(fake_data)
        
        # Contrastive divergence loss
        # Minimize energy for real data, maximize for fake data
        loss_real = energy_real.mean()
        loss_fake = -energy_fake.mean()
        loss_total = loss_real + loss_fake
        
        return {
            'total': loss_total,
            'real': loss_real,
            'fake': loss_fake
        }
    
    def train_step(self, real_data: Tensor) -> Dict[str, float]:
        """Single training step.
        
        Args:
            real_data: Batch of real data
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Generate fake data using Langevin dynamics
        with torch.no_grad():
            fake_data = self.model.langevin_sample(
                batch_size=real_data.size(0),
                steps=20,  # Fewer steps during training for efficiency
                step_size=0.01,
                noise_scale=0.01,
                device=self.device
            )
        
        # Compute loss
        losses = self.compute_loss(real_data, fake_data)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update statistics
        self.step += 1
        loss_dict = {k: v.item() for k, v in losses.items()}
        
        for key, value in loss_dict.items():
            self.losses[key].append(value)
        
        return loss_dict
    
    def generate_samples(
        self,
        num_samples: int = 64,
        steps: int = 100,
        step_size: float = 0.01
    ) -> Tensor:
        """Generate samples using Langevin dynamics.
        
        Args:
            num_samples: Number of samples to generate
            steps: Number of Langevin steps
            step_size: Step size for sampling
            
        Returns:
            Generated samples
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.langevin_sample(
                batch_size=num_samples,
                steps=steps,
                step_size=step_size,
                device=self.device
            )
