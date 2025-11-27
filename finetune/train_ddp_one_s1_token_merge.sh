#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B"
    --model_name "dove-s1"
    --model_type "real-sr"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s1-token-merge"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/HQ-VSR"
    --video_column "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/HQ-VSR.txt"
    --train_resolution "25x320x640"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1000
    --train_steps 10000
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 2e-5
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
    --stastic_frequency 500
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000
    --checkpointing_limit 3
    --resume_from_checkpoint "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s1-token-merge/checkpoint-6000"
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/"
    --validation_steps 500
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
    --degradation_config "/home/sig95vg/codes/DOVE/finetune/configs/degradation.yaml"
)

# Token Merge parameters
TOKEN_MERGE_WINDOW_SIZE=${TOKEN_MERGE_WINDOW_SIZE:-0}
TOKEN_MERGE_WINDOW_STRIDE=${TOKEN_MERGE_WINDOW_STRIDE:-1}
TOKEN_MERGE_RATIO_START=${TOKEN_MERGE_RATIO_START:-}
TOKEN_MERGE_RATIO_WARMUP_STEPS=${TOKEN_MERGE_RATIO_WARMUP_STEPS:-0}
TOKEN_MERGE_RATIO_SCHEDULE=${TOKEN_MERGE_RATIO_SCHEDULE:-linear}
TOKEN_MERGE_ARGS=(
    --enable_token_merge true
    --token_merge_seed "${TOKEN_MERGE_SEED:-42}"
    --token_merge_restore_adapter_expansion "${TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION:-2}"
    --token_merge_freeze_routes_only true
    --token_merge_window_size "${TOKEN_MERGE_WINDOW_SIZE}"
    --token_merge_window_stride "${TOKEN_MERGE_WINDOW_STRIDE}"
    --token_merge_ratio_warmup_steps "${TOKEN_MERGE_RATIO_WARMUP_STEPS}"
    --token_merge_ratio_schedule "${TOKEN_MERGE_RATIO_SCHEDULE}"
)

if [[ -n "${TOKEN_MERGE_ROUTES:-}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_routes "${TOKEN_MERGE_ROUTES}")
fi

if [[ -n "${TOKEN_MERGE_DEFAULT_RATIO:-}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_default_ratio "${TOKEN_MERGE_DEFAULT_RATIO}")
fi

if [[ -n "${TOKEN_MERGE_RATIO_START:-}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_ratio_start "${TOKEN_MERGE_RATIO_START}")
fi

# Resolve script directory for config reference
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

accelerate launch --config_file "${SCRIPT_DIR}/accelerate_config.yaml" "${SCRIPT_DIR}/train.py" \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${TOKEN_MERGE_ARGS[@]}"
