#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=uv_add_package
#SBATCH --output=uv_add_package.out
#SBATCH --nodelist=gorina8

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

set -euo pipefail

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"

# Ensure uv is on PATH
export PATH="${HOME}/.local/bin:${PATH}"

echo "uv: $(command -v uv)"
uv --version

source .finetune/bin/activate

# Install compatible PyTorch + torchvision + torchaudio
uv pip install scikit-learn==1.3.2


# Final verification
python -c "
import torch
import torchvision
import transformers
import unsloth
print('✓ PyTorch:', torch.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
"