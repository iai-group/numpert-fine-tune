#!/bin/bash
#SBATCH --job-name=inference_test_qwen3
#SBATCH --partition=gpu
##SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=12:00:00
#SBATCH --output=inference_test_qwen3.out
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9


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
uenv verbose cuda-12.5.1 cudnn-12.x-8.8.0
# Load environment
echo "Activating uv environment..."
echo $PATH
# source .finetune/bin/activate
uv sync
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=6

uv run python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
for repo in cache.repos:
    if 'Qwen3' in repo.repo_id:
        print(f'{repo.repo_id}: {repo.size_on_disk_str}, refs: {[r.refs for r in repo.revisions]}')
"

uv run python -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_OFFLINE'] = '1'
path = hf_hub_download('unsloth/Qwen3-8B', 'config.json')
print('Found config at:', path)
"

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Run training script
echo "=========================================="
echo "Starting Inference Script"
echo "=========================================="
echo ""

SCRIPT_DIR="/home/stud/u2929246/bhome/numpert-fine-tune"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""
    
bash ./scripts/test_scripts/inference_test.sh

# Print completion information
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
