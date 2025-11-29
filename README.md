# Energy-based Models for Image Generation

A production-ready implementation of Energy-based Models (EBMs) for image generation with comprehensive evaluation, interactive demos, and reproducible experiments.

## Overview

Energy-based Models learn an energy function E(x) that assigns lower energy to real data and higher energy to generated/fake data. This project implements:

- **Modern EBM Architecture**: Residual blocks with spectral normalization
- **Advanced Sampling**: Langevin dynamics and MCMC sampling
- **Comprehensive Evaluation**: FID, IS, LPIPS, and energy-based metrics
- **Interactive Demo**: Streamlit interface for real-time generation
- **Reproducible Experiments**: YAML-based configuration management

## Features

### Model Architecture
- Residual blocks with spectral normalization for stable training
- Adaptive architecture supporting different image sizes
- Proper energy function learning with contrastive divergence

### Sampling Methods
- **Langevin Dynamics**: Fast gradient-based sampling
- **MCMC**: Metropolis-Hastings sampling for theoretical soundness
- Configurable step sizes and noise scales

### Evaluation Metrics
- **FID**: Fréchet Inception Distance for quality assessment
- **IS**: Inception Score for diversity evaluation
- **LPIPS**: Learned perceptual similarity for diversity
- **Energy Statistics**: Real vs fake energy analysis

### Interactive Demo
- Real-time sample generation with parameter controls
- Multiple sampling methods and configurations
- Download generated samples
- Model comparison interface

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Energy-based-Models-for-Image-Generation.git
cd Energy-based-Models-for-Image-Generation

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Training

```bash
# Train on CIFAR-10 with default settings
python scripts/train.py --experiment cifar10_baseline

# Train on MNIST
python scripts/train.py --experiment mnist_baseline

# Train with custom config
python scripts/train.py --config configs/custom_config.yaml
```

### Sampling

```bash
# Generate samples from trained model
python scripts/sample.py --checkpoint assets/checkpoints/best_model.pth --num_samples 64

# Generate with specific parameters
python scripts/sample.py \
    --checkpoint assets/checkpoints/best_model.pth \
    --method langevin \
    --steps 200 \
    --step_size 0.005 \
    --num_samples 32
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
energy-based-models/
├── src/
│   ├── models/
│   │   └── ebm.py              # EBM model implementation
│   ├── data/
│   │   └── datasets.py         # Data loading and preprocessing
│   ├── utils/
│   │   └── metrics.py          # Evaluation metrics
│   └── configs/
│       └── config.py           # Configuration management
├── scripts/
│   ├── train.py                # Training script
│   └── sample.py               # Sampling script
├── configs/
│   ├── cifar10_baseline.yaml   # CIFAR-10 configuration
│   └── mnist_baseline.yaml     # MNIST configuration
├── demo/
│   └── app.py                  # Streamlit demo
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── assets/                     # Generated samples and checkpoints
└── docs/                       # Documentation
```

## Configuration

The project uses YAML-based configuration management with OmegaConf. Key configuration sections:

### Model Configuration
```yaml
model:
  in_channels: 3          # Input channels (1 for grayscale, 3 for RGB)
  base_channels: 64        # Base number of channels
  num_layers: 4            # Number of residual blocks
  image_size: 32          # Input image size
  spectral_norm: true     # Use spectral normalization
```

### Training Configuration
```yaml
training:
  batch_size: 64          # Batch size
  num_epochs: 100         # Number of training epochs
  learning_rate: 1e-4     # Learning rate
  beta1: 0.5             # Adam beta1
  beta2: 0.999           # Adam beta2
  weight_decay: 1e-5     # Weight decay
  gradient_clip: 1.0     # Gradient clipping
```

### Sampling Configuration
```yaml
sampling:
  method: langevin        # 'langevin' or 'mcmc'
  steps: 100             # Number of sampling steps
  step_size: 0.01        # Step size
  noise_scale: 0.01      # Noise scale (Langevin only)
  num_samples: 64        # Number of samples to generate
```

## Datasets

The project supports multiple datasets:

- **CIFAR-10**: 32x32 color images, 10 classes
- **MNIST**: 28x28 grayscale digits
- **Fashion-MNIST**: 28x28 grayscale clothing items
- **Toy Dataset**: Generated synthetic data for testing

### Dataset Usage

```python
from src.data.datasets import get_data_loaders

# Load CIFAR-10
train_loader, val_loader, test_loader = get_data_loaders(
    dataset_name='cifar10',
    batch_size=64,
    image_size=(32, 32)
)

# Load MNIST
train_loader, val_loader, test_loader = get_data_loaders(
    dataset_name='mnist',
    batch_size=128,
    image_size=(28, 28)
)
```

## Evaluation

### Metrics

The project provides comprehensive evaluation metrics:

1. **FID (Fréchet Inception Distance)**: Measures quality and diversity
2. **IS (Inception Score)**: Measures quality and diversity
3. **LPIPS Diversity**: Measures perceptual diversity
4. **Energy Statistics**: Real vs fake energy analysis

### Running Evaluation

```python
from src.utils.metrics import EBMMetrics

# Initialize metrics
metrics = EBMMetrics(device=device, num_samples=10000)

# Evaluate model
eval_results = metrics.evaluate_model(
    model=model,
    real_images=real_data,
    fake_images=generated_data
)

# Print results
metrics.print_metrics(eval_results)
```

## Advanced Usage

### Custom Model Architecture

```python
from src.models.ebm import EnergyBasedModel

# Create custom model
model = EnergyBasedModel(
    in_channels=3,
    base_channels=128,    # Larger model
    num_layers=6,         # More layers
    image_size=64         # Higher resolution
)
```

### Custom Sampling Parameters

```python
# Langevin dynamics with custom parameters
samples = model.langevin_sample(
    batch_size=32,
    steps=200,            # More steps for better quality
    step_size=0.005,      # Smaller step size
    noise_scale=0.005    # Less noise
)

# MCMC sampling
samples = model.mcmc_sample(
    batch_size=32,
    steps=200,
    step_size=0.01
)
```

### Training with Custom Configuration

```python
from src.configs.config import get_default_config, update_config

# Get default config
config = get_default_config()

# Update specific parameters
config = update_config(config, {
    'training': {
        'num_epochs': 200,
        'learning_rate': 5e-5
    },
    'sampling': {
        'steps': 150
    }
})
```

## Experiments

### Predefined Experiments

The project includes several predefined experiment configurations:

- `cifar10_baseline`: Standard CIFAR-10 training
- `mnist_baseline`: MNIST digit generation
- `fashion_mnist_baseline`: Fashion-MNIST clothing generation
- `toy_experiment`: Quick testing with synthetic data
- `high_resolution`: Higher resolution training
- `fast_training`: Quick training for experimentation

### Running Experiments

```bash
# Run predefined experiment
python scripts/train.py --experiment cifar10_baseline

# Run with custom overrides
python scripts/train.py \
    --experiment cifar10_baseline \
    --config configs/custom_overrides.yaml
```

## Model Cards

### CIFAR-10 Model
- **Dataset**: CIFAR-10 (50,000 training, 10,000 test images)
- **Resolution**: 32x32 RGB
- **Architecture**: 4-layer residual EBM with spectral normalization
- **Training**: 100 epochs, Adam optimizer, contrastive divergence loss
- **Performance**: FID ~50-80, IS ~6-8

### MNIST Model
- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Resolution**: 28x28 grayscale
- **Architecture**: 3-layer residual EBM
- **Training**: 50 epochs, smaller model for efficiency
- **Performance**: FID ~10-20, IS ~8-10

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable mixed precision training

2. **Poor Sample Quality**
   - Increase sampling steps
   - Adjust step size and noise scale
   - Train for more epochs

3. **Training Instability**
   - Use spectral normalization
   - Reduce learning rate
   - Enable gradient clipping

### Performance Tips

1. **Faster Training**
   - Use smaller model architecture
   - Reduce sampling steps during training
   - Use mixed precision

2. **Better Quality**
   - Increase model capacity
   - Use more sampling steps
   - Train for longer

3. **Memory Optimization**
   - Use gradient checkpointing
   - Reduce batch size
   - Use CPU offloading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Format code
black src/ scripts/
ruff check src/ scripts/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{energy_based_models,
  title={Energy-based Models for Image Generation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Energy-based-Models-for-Image-Generation}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The generative modeling community for research and insights
- Contributors and users who provide feedback and improvements
# Energy-based-Models-for-Image-Generation
