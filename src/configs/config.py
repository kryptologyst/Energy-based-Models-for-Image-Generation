"""Configuration management for Energy-based Model training and evaluation.

This module provides YAML-based configuration management with OmegaConf
for easy experimentation and reproducibility.
"""

from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os


def get_default_config() -> DictConfig:
    """Get default configuration for EBM training.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        # Model configuration
        'model': {
            'name': 'ebm',
            'in_channels': 3,
            'base_channels': 64,
            'num_layers': 4,
            'image_size': 32,
            'spectral_norm': True
        },
        
        # Training configuration
        'training': {
            'batch_size': 64,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'beta1': 0.5,
            'beta2': 0.999,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'save_every': 10,
            'eval_every': 5,
            'sample_every': 5
        },
        
        # Sampling configuration
        'sampling': {
            'method': 'langevin',  # 'langevin' or 'mcmc'
            'steps': 100,
            'step_size': 0.01,
            'noise_scale': 0.01,
            'num_samples': 64
        },
        
        # Data configuration
        'data': {
            'dataset': 'cifar10',  # 'cifar10', 'mnist', 'fashion_mnist', 'toy'
            'data_dir': './data',
            'num_workers': 4,
            'val_split': 0.1,
            'augment_prob': 0.5
        },
        
        # Evaluation configuration
        'evaluation': {
            'num_samples': 10000,
            'batch_size': 100,
            'compute_fid': True,
            'compute_is': True,
            'compute_lpips': True,
            'save_plots': True
        },
        
        # Logging and monitoring
        'logging': {
            'log_dir': './logs',
            'save_dir': './assets',
            'use_wandb': False,
            'use_tensorboard': True,
            'project_name': 'ebm-generation'
        },
        
        # Device and reproducibility
        'device': {
            'device': 'auto',  # 'auto', 'cuda', 'mps', 'cpu'
            'seed': 42,
            'deterministic': True
        },
        
        # Paths
        'paths': {
            'checkpoints': './assets/checkpoints',
            'samples': './assets/samples',
            'logs': './logs',
            'configs': './configs'
        }
    }
    
    return OmegaConf.create(config)


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        # Merge with defaults to ensure all keys exist
        default_config = get_default_config()
        config = OmegaConf.merge(default_config, config)
    else:
        config = get_default_config()
    
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def update_config(config: DictConfig, updates: Dict[str, Any]) -> DictConfig:
    """Update configuration with new values.
    
    Args:
        config: Current configuration
        updates: Dictionary of updates
        
    Returns:
        Updated configuration
    """
    return OmegaConf.merge(config, updates)


def get_device_config(config: DictConfig) -> str:
    """Get device configuration based on availability.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda', 'mps', 'cpu')
    """
    device_setting = config.device.device
    
    if device_setting == 'auto':
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_setting


def setup_reproducibility(config: DictConfig) -> None:
    """Setup reproducibility settings.
    
    Args:
        config: Configuration dictionary
    """
    import torch
    import numpy as np
    import random
    
    seed = config.device.seed
    deterministic = config.device.deterministic
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def create_experiment_config(
    base_config: DictConfig,
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """Create configuration for a specific experiment.
    
    Args:
        base_config: Base configuration
        experiment_name: Name of the experiment
        overrides: Configuration overrides
        
    Returns:
        Experiment-specific configuration
    """
    config = base_config.copy()
    
    # Add experiment name
    config.experiment_name = experiment_name
    
    # Apply overrides
    if overrides:
        config = update_config(config, overrides)
    
    # Update paths with experiment name
    config.paths.checkpoints = f"./assets/checkpoints/{experiment_name}"
    config.paths.samples = f"./assets/samples/{experiment_name}"
    config.paths.logs = f"./logs/{experiment_name}"
    
    return config


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    'cifar10_baseline': {
        'data': {'dataset': 'cifar10'},
        'model': {'image_size': 32, 'base_channels': 64},
        'training': {'num_epochs': 100, 'batch_size': 64}
    },
    
    'mnist_baseline': {
        'data': {'dataset': 'mnist'},
        'model': {'image_size': 28, 'in_channels': 1, 'base_channels': 32},
        'training': {'num_epochs': 50, 'batch_size': 128}
    },
    
    'fashion_mnist_baseline': {
        'data': {'dataset': 'fashion_mnist'},
        'model': {'image_size': 28, 'in_channels': 1, 'base_channels': 32},
        'training': {'num_epochs': 50, 'batch_size': 128}
    },
    
    'toy_experiment': {
        'data': {'dataset': 'toy'},
        'model': {'image_size': 32, 'base_channels': 32},
        'training': {'num_epochs': 20, 'batch_size': 32}
    },
    
    'high_resolution': {
        'data': {'dataset': 'cifar10'},
        'model': {'image_size': 64, 'base_channels': 128, 'num_layers': 6},
        'training': {'num_epochs': 200, 'batch_size': 32},
        'sampling': {'steps': 200}
    },
    
    'fast_training': {
        'data': {'dataset': 'cifar10'},
        'model': {'base_channels': 32, 'num_layers': 2},
        'training': {'num_epochs': 20, 'batch_size': 128},
        'sampling': {'steps': 50}
    }
}


def get_experiment_config(experiment_name: str) -> DictConfig:
    """Get predefined experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment configuration
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {available}")
    
    base_config = get_default_config()
    experiment_overrides = EXPERIMENT_CONFIGS[experiment_name]
    
    return create_experiment_config(base_config, experiment_name, experiment_overrides)
