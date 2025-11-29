"""Unit tests for Energy-based Model components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ebm import EnergyBasedModel, EBMTrainer, ResidualBlock
from data.datasets import get_data_loaders, create_toy_dataset
from utils.metrics import EBMMetrics
from configs.config import get_default_config, get_device_config, setup_reproducibility


class TestResidualBlock:
    """Test ResidualBlock implementation."""
    
    def test_residual_block_forward(self):
        """Test residual block forward pass."""
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(4, 64, 32, 32)
        output = block(x)
        
        assert output.shape == x.shape
        assert torch.is_tensor(output)
    
    def test_residual_block_stride(self):
        """Test residual block with stride."""
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(4, 64, 32, 32)
        output = block(x)
        
        assert output.shape == (4, 128, 16, 16)
    
    def test_residual_block_channels(self):
        """Test residual block with different input/output channels."""
        block = ResidualBlock(32, 64, stride=1)
        x = torch.randn(4, 32, 16, 16)
        output = block(x)
        
        assert output.shape == (4, 64, 16, 16)


class TestEnergyBasedModel:
    """Test EnergyBasedModel implementation."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = EnergyBasedModel(
            in_channels=3,
            base_channels=64,
            num_layers=4,
            image_size=32
        )
        
        assert isinstance(model, nn.Module)
        assert model.in_channels == 3
        assert model.base_channels == 64
        assert model.num_layers == 4
        assert model.image_size == 32
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = EnergyBasedModel(image_size=32)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (4, 1)
        assert torch.is_tensor(output)
    
    def test_model_energy(self):
        """Test energy computation."""
        model = EnergyBasedModel(image_size=32)
        x = torch.randn(4, 3, 32, 32)
        energy = model.energy(x)
        
        assert energy.shape == (4, 1)
        assert torch.is_tensor(energy)
    
    def test_langevin_sampling(self):
        """Test Langevin dynamics sampling."""
        model = EnergyBasedModel(image_size=32)
        device = torch.device('cpu')
        
        samples = model.langevin_sample(
            batch_size=4,
            steps=10,
            step_size=0.01,
            noise_scale=0.01,
            device=device
        )
        
        assert samples.shape == (4, 3, 32, 32)
        assert torch.is_tensor(samples)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0
    
    def test_mcmc_sampling(self):
        """Test MCMC sampling."""
        model = EnergyBasedModel(image_size=32)
        device = torch.device('cpu')
        
        samples = model.mcmc_sample(
            batch_size=4,
            steps=10,
            step_size=0.01,
            device=device
        )
        
        assert samples.shape == (4, 3, 32, 32)
        assert torch.is_tensor(samples)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0


class TestEBMTrainer:
    """Test EBMTrainer implementation."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = EnergyBasedModel()
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device)
        
        assert trainer.model == model
        assert trainer.device == device
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_compute_loss(self):
        """Test loss computation."""
        model = EnergyBasedModel()
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device)
        
        real_data = torch.randn(4, 3, 32, 32)
        fake_data = torch.randn(4, 3, 32, 32)
        
        losses = trainer.compute_loss(real_data, fake_data)
        
        assert 'total' in losses
        assert 'real' in losses
        assert 'fake' in losses
        assert all(torch.is_tensor(loss) for loss in losses.values())
    
    def test_train_step(self):
        """Test single training step."""
        model = EnergyBasedModel()
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device)
        
        real_data = torch.randn(4, 3, 32, 32)
        
        losses = trainer.train_step(real_data)
        
        assert 'total' in losses
        assert 'real' in losses
        assert 'fake' in losses
        assert all(isinstance(loss, float) for loss in losses.values())
    
    def test_generate_samples(self):
        """Test sample generation."""
        model = EnergyBasedModel()
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device)
        
        samples = trainer.generate_samples(
            num_samples=4,
            steps=10,
            step_size=0.01
        )
        
        assert samples.shape == (4, 3, 32, 32)
        assert torch.is_tensor(samples)


class TestDataLoaders:
    """Test data loading functionality."""
    
    def test_cifar10_loaders(self):
        """Test CIFAR-10 data loaders."""
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name='cifar10',
            batch_size=32,
            num_workers=0,  # Use 0 for testing
            image_size=(32, 32)
        )
        
        # Test loader shapes
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == 32  # batch size
        assert images.shape[1] == 3   # channels
        assert images.shape[2] == 32  # height
        assert images.shape[3] == 32  # width
        assert images.min() >= -1.0
        assert images.max() <= 1.0
    
    def test_mnist_loaders(self):
        """Test MNIST data loaders."""
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name='mnist',
            batch_size=32,
            num_workers=0,
            image_size=(28, 28)
        )
        
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == 32
        assert images.shape[1] == 1   # grayscale
        assert images.shape[2] == 28
        assert images.shape[3] == 28
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_toy_dataset(
                num_samples=100,
                image_size=(32, 32),
                num_classes=5,
                save_dir=temp_dir
            )
            
            # Check if files were created
            toy_path = Path(temp_dir)
            assert toy_path.exists()
            
            # Check class directories
            for i in range(5):
                class_dir = toy_path / f'class_{i:02d}'
                assert class_dir.exists()
                assert len(list(class_dir.glob('*.npy'))) == 100


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        device = torch.device('cpu')
        metrics = EBMMetrics(device, num_samples=100, batch_size=10)
        
        assert metrics.device == device
        assert metrics.num_samples == 100
        assert metrics.batch_size == 10
    
    def test_energy_statistics(self):
        """Test energy statistics computation."""
        model = EnergyBasedModel()
        device = torch.device('cpu')
        metrics = EBMMetrics(device)
        
        real_images = torch.randn(10, 3, 32, 32)
        fake_images = torch.randn(10, 3, 32, 32)
        
        stats = metrics.compute_energy_statistics(model, real_images, fake_images)
        
        assert 'energy_real' in stats
        assert 'energy_fake' in stats
        assert 'energy_gap' in stats
        assert all(isinstance(value, float) for value in stats.values())


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        
        assert 'model' in config
        assert 'training' in config
        assert 'sampling' in config
        assert 'data' in config
        assert 'evaluation' in config
    
    def test_device_config(self):
        """Test device configuration."""
        config = get_default_config()
        device = get_device_config(config)
        
        assert device in ['cuda', 'mps', 'cpu']
    
    def test_reproducibility_setup(self):
        """Test reproducibility setup."""
        config = get_default_config()
        
        # This should not raise an error
        setup_reproducibility(config)


class TestIntegration:
    """Integration tests."""
    
    def test_training_integration(self):
        """Test basic training integration."""
        # Create small model for testing
        model = EnergyBasedModel(
            in_channels=1,
            base_channels=16,
            num_layers=2,
            image_size=16
        )
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device, lr=1e-3)
        
        # Create small dataset
        train_loader, _, _ = get_data_loaders(
            dataset_name='mnist',
            batch_size=8,
            num_workers=0,
            image_size=(16, 16)
        )
        
        # Run a few training steps
        for i, (real_data, _) in enumerate(train_loader):
            if i >= 3:  # Only test a few steps
                break
            
            losses = trainer.train_step(real_data)
            assert 'total' in losses
            assert isinstance(losses['total'], float)
    
    def test_sampling_integration(self):
        """Test sampling integration."""
        model = EnergyBasedModel(image_size=16)
        device = torch.device('cpu')
        trainer = EBMTrainer(model, device)
        
        # Generate samples
        samples = trainer.generate_samples(
            num_samples=4,
            steps=10,
            step_size=0.01
        )
        
        assert samples.shape == (4, 3, 16, 16)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
