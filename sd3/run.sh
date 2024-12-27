#!/bin/bash

# Clear terminal
clear

# Python env vars
export PYTHONWARNINGS="ignore"

# Apple Silicon specific environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
export MPS_ALLOCATOR_RELEASE_INTERVAL=100


accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
  --instance_data_dir="../JimSantora" \
  --output_dir="../trained-sd3.5-jimsantora" \
  --mixed_precision="no" \
  --instance_prompt="a photo of Jim Santora" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --seed="0" \
  --gradient_checkpointing \
  --cache_latents \
  --use_txt_as_prompts

  #--validation_prompt="A photo of Jim Santora in a chair" \
  #--validation_epochs=100 \
  #--num_validation_images=1 \
