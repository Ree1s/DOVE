#!/usr/bin/env bash

DOVER_WEIGHTS=/data/42-julia-hpc-rz-cv/sig95vg/checkpoints/DOVER.pth

# UDM10
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/UDM10 \
    --is_vae_st \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/UDM10 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/UDM10 \
    --out /home/sig95vg/codes/DOVE/results/DOVE/UDM10 \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/UDM10 \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/UDM10

# SPMCS
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/SPMCS/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/SPMCS \
    --is_vae_st \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/SPMCS/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/SPMCS \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/SPMCS \
    --out /home/sig95vg/codes/DOVE/results/DOVE/SPMCS \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/SPMCS \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/SPMCS

# YouHQ40
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/YouHQ40/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40 \
    --is_vae_st \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/YouHQ/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40 \
    --out /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40 \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40 \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/YouHQ40

# RealVSR
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/RealVSR/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/RealVSR \
    --is_vae_st \
    --upscale 1 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/RealVSR/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/RealVSR \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/RealVSR \
    --out /home/sig95vg/codes/DOVE/results/DOVE/RealVSR \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/RealVSR \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/RealVSR

# MVSR4x
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/MVSR4x/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x \
    --is_vae_st \
    --upscale 1 \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/MVSR4x/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x \
    --metrics psnr,ssim,lpips,dists,clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x \
    --out /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/MVSR4x

# VideoLQ
python inference_script.py \
    --input_dir /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/VideoLQ/LQ-Video \
    --model_path /data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE \
    --output_path /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ \
    --is_vae_st \

python eval_metrics.py \
    --gt /data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/VideoLQ/GT \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ \
    --metrics clipiqa

# DOVER temporal consistency metric
python finetune/scripts/eval_dover.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ \
    --out /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ \
    --weights ${DOVER_WEIGHTS}

# Ewarp temporal warping error
python finetune/scripts/eval_ewarp.py \
    --pred /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ \
    --metric warping_error \
    --model /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/raft-things.pth \
    --out /home/sig95vg/codes/DOVE/results/DOVE/VideoLQ
