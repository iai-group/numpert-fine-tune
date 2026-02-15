#!/bin/bash
#SBATCH --job-name=finetune_unsloth
#SBATCH --partition=gpu
##  --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=12:00:00
#SBATCH --output=finetune_unsloth.out
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

# Load environment
echo "Activating uv environment..."
echo $PATH
source .finetune/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=6


# Verify GPU is available
echo ""
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Run training script
echo "=========================================="
echo "Starting Fine-tuning Script"
echo "=========================================="
echo ""


echo "Working directory: $(pwd)"
echo ""
    
bash ./scripts/finetune_qwen.sh

# Print completion information
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
