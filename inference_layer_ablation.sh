#!/usr/bin/env bash
set -euo pipefail

# Quick layer-importance sweep for CogVideoX 1.5 style transformer (42 blocks).
# Configurable via env vars so you can point at different datasets/checkpoints.

MODEL_PATH=${MODEL_PATH:-/data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE}
INPUT_DIR=${INPUT_DIR:-/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/LQ-Video}
GT_DIR=${GT_DIR:-/data/42-julia-hpc-rz-cv/sig95vg/DOVE/datasets/test/UDM10/GT}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/sig95vg/codes/DOVE/results/layer_ablation}
EVAL_METRICS=${EVAL_METRICS:-psnr,ssim,lpips,dists,clipiqa}
FPS=${FPS:-8}
DTYPE=${DTYPE:-bfloat16}
NUM_LAYERS=${NUM_LAYERS:-42}  # CogVideoX 1.5 uses 42 layers (0-41)

mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --input_dir "${INPUT_DIR}"
  --gt_dir "${GT_DIR}"
  --model_path "${MODEL_PATH}"
  --output_path "${OUTPUT_ROOT}/original"
  --fps "${FPS}"
  --dtype "${DTYPE}"
  --is_vae_st
  --eval_metrics "${EVAL_METRICS}"
)

echo "[baseline] writing to ${OUTPUT_ROOT}/original"
python inference_script.py "${COMMON_ARGS[@]}"

for LAYER in $(seq 0 $((NUM_LAYERS - 1))); do
  OUT_DIR="${OUTPUT_ROOT}/skip_layer_$(printf "%02d" "${LAYER}")"
  echo "[skip layer ${LAYER}] writing to ${OUT_DIR}"
  python inference_script.py "${COMMON_ARGS[@]}" --output_path "${OUT_DIR}" --skip_layers "${LAYER}"
done
