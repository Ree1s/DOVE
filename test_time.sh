#!/usr/bin/env bash

set -euo pipefail

echo "=== Profiling baseline DOVE transformer ==="
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/dove_time/UDM10 \
    --is_vae_st \
    --profile_transformer \
    --eval_metrics '' \

echo "=== Profiling token-merge DOVE transformer ==="
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_learnable_10_17_28_35_ratio0.36_otherfrozen_lr5e-6_slidewindow3_1_woema/ckpt-400/-sft \
    --output_path /home/sig95vg/codes/DOVE/results/dove_merge_time/UDM10 \
    --is_vae_st \
    --token_merge_routes "10-17@0.36;28-35@0.36" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \
    --profile_transformer \
    --eval_metrics ''
