"""Setup script for Energy-based Models project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="energy-based-models",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modern Energy-based Models for image generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/energy-based-models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pre-commit>=3.0.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ebm-train=scripts.train:main",
            "ebm-sample=scripts.sample:main",
        ],
    },
)
