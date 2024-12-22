from setuptools import setup, find_packages

setup(
    name="yem",
    version="0.1.0",
    author="Jim Santora",
    author_email="jim.santora+gh@gmail.com",
    description="Stable Diffusion 3 fine-tuning tool optimized for Apple Silicon",
    long_description="""
    Yem is a specialized tool for fine-tuning Stable Diffusion 3 models
    on Apple Silicon hardware using LoRA. It features automatic memory
    management, thermal monitoring, and M-series chip optimizations.
    """,
    url="https://github.com/jimsantora/yem",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "accelerate>=0.27.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.19.0",
        "sentencepiece>=0.1.99",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yem=yem.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yem": ["config.yaml"],
    },
)