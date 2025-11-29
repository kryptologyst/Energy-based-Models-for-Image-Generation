#!/usr/bin/env python3
"""Quick test script to verify the EBM implementation works."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work."""
    try:
        from models.ebm import EnergyBasedModel, EBMTrainer
        from data.datasets import get_data_loaders
        from utils.metrics import EBMMetrics
        from configs.config import get_default_config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from models.ebm import EnergyBasedModel
        
        model = EnergyBasedModel(
            in_channels=1,
            base_channels=16,
            num_layers=2,
            image_size=16
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_forward_pass():
    """Test forward pass."""
    try:
        from models.ebm import EnergyBasedModel
        
        model = EnergyBasedModel(
            in_channels=1,
            base_channels=16,
            num_layers=2,
            image_size=16
        )
        
        x = torch.randn(2, 1, 16, 16)
        output = model(x)
        
        assert output.shape == (2, 1)
        print("✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        return False

def test_sampling():
    """Test sampling methods."""
    try:
        from models.ebm import EnergyBasedModel
        
        model = EnergyBasedModel(
            in_channels=1,
            base_channels=16,
            num_layers=2,
            image_size=16
        )
        device = torch.device('cpu')
        
        # Test Langevin sampling
        samples = model.langevin_sample(
            batch_size=2,
            steps=10,
            step_size=0.01,
            device=device
        )
        
        assert samples.shape == (2, 1, 16, 16)
        print("✓ Langevin sampling successful")
        
        # Test MCMC sampling
        samples = model.mcmc_sample(
            batch_size=2,
            steps=10,
            step_size=0.01,
            device=device
        )
        
        assert samples.shape == (2, 1, 16, 16)
        print("✓ MCMC sampling successful")
        
        return True
    except Exception as e:
        print(f"✗ Sampling error: {e}")
        return False

def test_data_loading():
    """Test data loading."""
    try:
        from data.datasets import get_data_loaders
        
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name='mnist',
            batch_size=4,
            num_workers=0,
            image_size=(16, 16)
        )
        
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == 4
        assert images.shape[1] == 1
        assert images.shape[2] == 16
        assert images.shape[3] == 16
        
        print("✓ Data loading successful")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_config():
    """Test configuration management."""
    try:
        from configs.config import get_default_config, get_device_config
        
        config = get_default_config()
        device = get_device_config(config)
        
        assert device in ['cuda', 'mps', 'cpu']
        print("✓ Configuration management successful")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Energy-based Model implementation...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_sampling,
        test_data_loading,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
