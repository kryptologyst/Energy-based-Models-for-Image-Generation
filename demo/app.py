"""Streamlit demo for Energy-based Model generation.

This demo provides an interactive interface for generating samples
from trained EBM models with various parameters.
"""

import streamlit as st
import torch
import torchvision
import numpy as np
from pathlib import Path
import sys
import os
from PIL import Image
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ebm import EnergyBasedModel
from configs.config import load_config, get_device_config


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load model with caching."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        model = EnergyBasedModel()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def generate_samples(
    model: EnergyBasedModel,
    device: torch.device,
    num_samples: int,
    method: str,
    steps: int,
    step_size: float,
    noise_scale: float
) -> torch.Tensor:
    """Generate samples."""
    with torch.no_grad():
        if method == 'Langevin Dynamics':
            samples = model.langevin_sample(
                batch_size=num_samples,
                steps=steps,
                step_size=step_size,
                noise_scale=noise_scale,
                device=device
            )
        elif method == 'MCMC':
            samples = model.mcmc_sample(
                batch_size=num_samples,
                steps=steps,
                step_size=step_size,
                device=device
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return samples


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    if tensor.dim() == 4:
        # Batch of images
        grid = torchvision.utils.make_grid(tensor, nrow=8, padding=2)
        image = torchvision.transforms.ToPILImage()(grid)
    else:
        # Single image
        image = torchvision.transforms.ToPILImage()(tensor)
    
    return image


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Energy-based Model Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Energy-based Model Demo")
    st.markdown("Generate images using trained Energy-based Models")
    
    # Sidebar for controls
    st.sidebar.header("Model Settings")
    
    # Model selection
    checkpoint_dir = Path("./assets/checkpoints")
    available_checkpoints = []
    
    if checkpoint_dir.exists():
        available_checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not available_checkpoints:
        st.error("No trained models found! Please train a model first.")
        st.stop()
    
    checkpoint_names = [cp.name for cp in available_checkpoints]
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model",
        checkpoint_names,
        help="Choose a trained model checkpoint"
    )
    
    # Load model
    checkpoint_path = checkpoint_dir / selected_checkpoint
    
    try:
        model, device = load_model(str(checkpoint_path))
        st.sidebar.success(f"Model loaded: {selected_checkpoint}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=64,
        value=16,
        step=1
    )
    
    method = st.sidebar.selectbox(
        "Sampling Method",
        ["Langevin Dynamics", "MCMC"],
        help="Choose the sampling method"
    )
    
    steps = st.sidebar.slider(
        "Sampling Steps",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="More steps = better quality but slower"
    )
    
    step_size = st.sidebar.slider(
        "Step Size",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Step size for sampling"
    )
    
    if method == "Langevin Dynamics":
        noise_scale = st.sidebar.slider(
            "Noise Scale",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Noise scale for Langevin dynamics"
        )
    else:
        noise_scale = 0.01  # Not used for MCMC
    
    # Seed control
    st.sidebar.header("Randomness")
    use_seed = st.sidebar.checkbox("Use Fixed Seed", value=False)
    
    if use_seed:
        seed = st.sidebar.number_input(
            "Seed",
            min_value=0,
            max_value=2**32-1,
            value=42,
            step=1
        )
    else:
        seed = None
    
    # Generate button
    if st.sidebar.button("🎨 Generate Samples", type="primary"):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        with st.spinner("Generating samples..."):
            try:
                samples = generate_samples(
                    model=model,
                    device=device,
                    num_samples=num_samples,
                    method=method,
                    steps=steps,
                    step_size=step_size,
                    noise_scale=noise_scale
                )
                
                # Convert to PIL image
                image = tensor_to_pil(samples)
                
                # Display image
                st.image(image, caption=f"Generated Samples ({method}, {steps} steps)")
                
                # Download button
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Image",
                    data=img_buffer.getvalue(),
                    file_name=f"ebm_samples_{method.lower()}_{steps}steps.png",
                    mime="image/png"
                )
                
                # Show generation info
                st.info(f"""
                **Generation Info:**
                - Method: {method}
                - Steps: {steps}
                - Step Size: {step_size}
                - Samples: {num_samples}
                - Device: {device}
                """)
                
            except Exception as e:
                st.error(f"Error generating samples: {e}")
    
    # Model info
    st.sidebar.header("Model Info")
    st.sidebar.text(f"Device: {device}")
    st.sidebar.text(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Instructions
    st.markdown("""
    ## Instructions
    
    1. **Select a Model**: Choose a trained model checkpoint from the dropdown
    2. **Adjust Parameters**: Use the sliders to control generation parameters
    3. **Generate**: Click the generate button to create new samples
    4. **Download**: Save the generated images to your computer
    
    ### Parameter Guide
    
    - **Sampling Steps**: More steps generally produce better quality but take longer
    - **Step Size**: Controls how much the samples change at each step
    - **Noise Scale**: Only for Langevin dynamics, controls randomness
    - **Method**: 
        - Langevin Dynamics: Faster, good for most cases
        - MCMC: Slower but more theoretically sound
    
    ### Tips
    
    - Start with default parameters and adjust based on results
    - Use fixed seeds for reproducible results
    - Higher step counts improve quality but increase generation time
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with PyTorch and Streamlit")


if __name__ == "__main__":
    main()
