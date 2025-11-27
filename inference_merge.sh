#!/usr/bin/env bash

DOVER_WEIGHTS=/data/42-julia-hpc-rz-cv/sig95vg/checkpoints/DOVER.pth

# UDM10
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10 \
    --is_vae_st \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10 \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10 \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10 \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge_distillation_dynamic_26_34_22_25/UDM10

# SPMCS
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/SPMCS/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS \
    --is_vae_st \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/SPMCS/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/SPMCS

# YouHQ40
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/YouHQ40/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40 \
    --is_vae_st \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/YouHQ/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40 \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40 \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40 \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/YouHQ40

# RealVSR
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/RealVSR/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR \
    --is_vae_st \
    --upscale 1 \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/RealVSR/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/RealVSR

# MVSR4x
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/MVSR4x/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x \
    --is_vae_st \
    --upscale 1 \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/MVSR4x/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/MVSR4x

# VideoLQ
python inference_script_merge.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/VideoLQ/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/checkpoint/DOVE-s2-temporal_merge_ablation_routes_22_25_26_34_lr5e-6_slidewindow3_1_woema/ckpt-3400-sft \
    --output_path /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ \
    --is_vae_st \
    --token_merge_routes "26-34@0.34;22-25@0.18" \
    --token_merge_window_size 3 \
    --token_merge_window_stride 1 \
    --token_merge_seed 42 \
    --token_merge_restore_adapter_expansion 2 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/VideoLQ/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ \
    --metrics clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOV_merge/VideoLQ
