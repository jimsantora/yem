#!/bin/bash

# Clear terminal
clear

# Python env vars
export PYTHONWARNINGS="ignore"

# Apple Silicon specific environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
export MPS_ALLOCATOR_RELEASE_INTERVAL=100

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --instance_data_dir="../JimSantora" \
  --output_dir="../trained-flux-jimsantora" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --mixed_precision="fp16" \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --guidance_scale=1 \
  --report_to="wandb" \
  --cache_latents \
  --gradient_checkpointing \
  --seed="0"