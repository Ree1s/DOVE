#!/bin/bash
#SBATCH -J dove_stage2_tm        # Job name
#SBATCH -c 32                     # Number of CPU cores
#SBATCH --mem=512G                # Total memory
#SBATCH -p h100                   # GPU partition (adjust as needed)
#SBATCH --gres=gpu:4            # Number of GPUs
#SBATCH --tmp=20G                 # Local scratch space
#SBATCH --mail-type=ALL           # Job begin/end/fail notifications
#SBATCH --mail-user=sicheng.gao@uni-wuerzburg.de
#SBATCH --output=logs/dove_s2_temporalmerge_learnable_otherfrozen_lr5e-6_slidewindow3_1_woema_%j.out
#SBATCH --error=logs/dove_s2_temporamerge_learnable_otherfrozen_lr5e-6_slidewindow3_1_woema_%j.err

set -euo pipefail

mkdir -p logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

CONDA_BASE="/home/sig95vg/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate DOVE

which python
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

cd /home/sig95vg/codes/DOVE/finetune

# Default token-merge schedule if none provided
export TOKEN_MERGE_ROUTES=${TOKEN_MERGE_ROUTES:-"10-17@0.64;28-35@0.64"}
export TOKEN_MERGE_DEFAULT_RATIO=${TOKEN_MERGE_DEFAULT_RATIO:-}
export TOKEN_MERGE_SEED=${TOKEN_MERGE_SEED:-42}
export TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION=${TOKEN_MERGE_RESTORE_ADAPTER_EXPANSION:-2}
export TOKEN_MERGE_WINDOW_SIZE=${TOKEN_MERGE_WINDOW_SIZE:-3}
export TOKEN_MERGE_WINDOW_STRIDE=${TOKEN_MERGE_WINDOW_STRIDE:-1}

bash ./train_ddp_one_s2_token_merge.sh

echo "Job finished at $(date)"
