#!/bin/bash
#SBATCH -J dove_stage1             # Job name
#SBATCH -c 32                      # Number of CPU cores
#SBATCH --mem=512G                 # Total memory
#SBATCH -p h100                    # GPU partition (adjust to your cluster)
#SBATCH --gres=gpu:4               # Number of GPUs
#SBATCH --tmp=20G                  # Local scratch space
#SBATCH --mail-type=ALL            # Send email on job begin/end/fail
#SBATCH --mail-user=sicheng.gao@uni-wuerzburg.de  # TODO: replace with your email
#SBATCH --output=logs/dove_s1_%j.out
#SBATCH --error=logs/dove_s1_%j.err

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

cd /home/sig95vg/codes/DOVE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=72000
export NCCL_DEBUG=INFO

echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

cd finetune
bash ./train_ddp_one_s1.sh

echo "Job finished at $(date)"
