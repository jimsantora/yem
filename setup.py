from setuptools import setup, find_packages

setup(
    name="yem",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
    entry_points={
        "console_scripts": [
            "yem=yem.main:main",
        ],
    },
)
