#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Default hyperparameters (mirrors HPC launcher so this script is self-contained)
TOKEN_MERGE_ROUTES="${TOKEN_MERGE_ROUTES:-10-17@0.36;28-35@0.36}"
TOKEN_MERGE_DEFAULT_RATIO="${TOKEN_MERGE_DEFAULT_RATIO:-}"
TOKEN_MERGE_SEED="${TOKEN_MERGE_SEED:-42}"
TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION="${TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION:-2}"
TOKEN_MERGE_WINDOW_SIZE="${TOKEN_MERGE_WINDOW_SIZE:-3}"
TOKEN_MERGE_WINDOW_STRIDE="${TOKEN_MERGE_WINDOW_STRIDE:-1}"

ENABLE_RELATIONAL_KD="${ENABLE_RELATIONAL_KD:-false}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE/}"
RELATIONAL_KD_WEIGHT="${RELATIONAL_KD_WEIGHT:-0.0}"
RELATIONAL_KD_LAYERS="${RELATIONAL_KD_LAYERS:-}"
ENABLE_TEACHER_LPIPS_KD="${ENABLE_TEACHER_LPIPS_KD:-true}"
TEACHER_LPIPS_WEIGHT="${TEACHER_LPIPS_WEIGHT:-0.2}"
TOKEN_MERGE_RATIO_START="${TOKEN_MERGE_RATIO_START:-0.1}"
TOKEN_MERGE_RATIO_WARMUP_STEPS="${TOKEN_MERGE_RATIO_WARMUP_STEPS:-500}"
TOKEN_MERGE_RATIO_SCHEDULE="${TOKEN_MERGE_RATIO_SCHEDULE:-cosine}"

# Model Configuration
MODEL_ARGS=(
    --model_path "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE/"
    --model_name "dove-s2"
    --model_type "real-sr-image-video"
    --training_type "sft"
)

# LORA_ARGS=(
#     --rank 64
#     --lora_alpha 64
# )

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_learnable_10_17_28_35_ratio0.64_lr5e-6_slidewindow3_1_woema_kd_1500iterations/"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/HQ-VSR"
    --video_column "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/HQ-VSR.txt"
    --image_data_root "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/DIV2K_train_HR"
    --image_column "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/train/DIV2K_train_HR.txt"
    --train_resolution "2x320x640"  # (frames x height x width), frames should be 8N+1
    # --crop_mode "resize_random_crop"
    --image_ratio 0.8
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --train_steps 1500
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 4
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --learning_rate 3e-6
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"  # ["constant_with_warmup", "decay_with_warmup"]
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
    --stastic_frequency 100
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100 # save checkpoint every x steps
    --checkpointing_limit 3 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --resume_from_checkpoint "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_learnable_10_17_28_35_ratio0.64_lr5e-6_slidewindow3_1_woema_kd_1500iterations/checkpoint-1300"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10"
    --validation_steps 300  # should be multiple of checkpointing_steps
    --validation_videos "LQ-Video.txt"
    --validation_ref_videos "GT-Video.txt"
    # --validation_prompts "prompts.txt"
    --gen_fps 8
    --raw_test true
    --num_inference_steps 1
    --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"  # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']
)

# SR parameters
SR_ARGS=(
    --is_latent false
    --is_cache true
    --empty_prompt true
    --prompt_cache "prompt_embeddings"
    --sr_noise_step 399
    --noise_step 0
    --degradation_config "/home/sig95vg/codes/DOVE/finetune/configs/degradation_image_video.yaml"
)

# Perceptual Loss parameters
Per_ARGS=(
    --use_perceptual_loss true
    --dists_weight 1.0
    --frame_diff_weight 1.0
)

# Token Merge parameters (enabled by default; adjust as needed)
TOKEN_MERGE_ARGS=(
    --enable_token_merge true
    --token_merge_seed "${TOKEN_MERGE_SEED}"
    --token_merge_restore_adapter_expansion "${TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION}"
    --token_merge_freeze_routes_only true
    --token_merge_window_size "${TOKEN_MERGE_WINDOW_SIZE}"
    --token_merge_window_stride "${TOKEN_MERGE_WINDOW_STRIDE}"
    --token_merge_ratio_warmup_steps "${TOKEN_MERGE_RATIO_WARMUP_STEPS}"
    --token_merge_ratio_schedule "${TOKEN_MERGE_RATIO_SCHEDULE}"
)

if [[ -n "${TOKEN_MERGE_ROUTES}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_routes "${TOKEN_MERGE_ROUTES}")
fi

if [[ -n "${TOKEN_MERGE_DEFAULT_RATIO}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_default_ratio "${TOKEN_MERGE_DEFAULT_RATIO}")
fi

if [[ -n "${TOKEN_MERGE_RATIO_START}" ]]; then
    TOKEN_MERGE_ARGS+=(--token_merge_ratio_start "${TOKEN_MERGE_RATIO_START}")
fi

# Relational KD parameters (optional)
KD_ARGS=(
    --enable_relational_kd "${ENABLE_RELATIONAL_KD}"
    --relational_kd_weight "${RELATIONAL_KD_WEIGHT}"
    --enable_teacher_lpips_kd "${ENABLE_TEACHER_LPIPS_KD}"
    --teacher_lpips_weight "${TEACHER_LPIPS_WEIGHT}"
)
if [[ -n "${TEACHER_MODEL_PATH}" ]]; then
    KD_ARGS+=(--teacher_model_path "${TEACHER_MODEL_PATH}")
fi
if [[ -n "${RELATIONAL_KD_LAYERS}" ]]; then
    KD_ARGS+=(--relational_kd_layers "${RELATIONAL_KD_LAYERS}")
fi

# Resolve script directory for config reference
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Combine all arguments and launch training with token merge
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
    "${Per_ARGS[@]}" \
    "${TOKEN_MERGE_ARGS[@]}" \
    "${KD_ARGS[@]}"
