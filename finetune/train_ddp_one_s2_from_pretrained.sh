#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Path to downloaded DOVE stage-2 weights (diffusers format)
DOVE_STAGE2_DIR="../pretrained_models/DOVE"

# Model Configuration
MODEL_ARGS=(
    --model_path "${DOVE_STAGE2_DIR}"
    --model_name "dove-s2"
    --model_type "real-sr-image-video"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-s2-from-pretrained"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train"
    --video_column "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/HQ-VSR.txt"
    --image_data_root "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train"
    --image_column "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/DIV2K_train_HR.txt"
    --train_resolution "2x320x640"
    --image_ratio 0.8
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10
    --train_steps 500
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 5e-6
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
    --stastic_frequency 100
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100
    --checkpointing_limit 3
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10"
    --validation_steps 100
    --validation_videos "LQ-Video.txt"
    --validation_ref_videos "GT-Video.txt"
    --gen_fps 8
    --raw_test true
    --num_inference_steps 1
    --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"
)

# SR parameters
SR_ARGS=(
    --is_latent false
    --is_cache true
    --empty_prompt true
    --prompt_cache "prompt_embeddings"
    --sr_noise_step 399
    --noise_step 0
    --degradation_config "configs/degradation_image_video.yaml"
)

# Perceptual Loss parameters
Per_ARGS=(
    --use_perceptual_loss true
    --dists_weight 1.0
    --frame_diff_weight 1.0
)

accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${Per_ARGS[@]}"
