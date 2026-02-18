#!/bin/bash
#SBATCH --job-name=qwen3-8b-finetune
#SBATCH --partition=gpu
##  --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=24:00:00
#SBATCH --output=scripts/slurm_out/train/qwen3-8b-finetune_enhanced_batch.out
#SBATCH --mem=300G
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

# Load environment
echo "Activating uv environment..."
#echo $PATH
# source .finetune/bin/activate


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
    
bash ./scripts/train/model_configs/qwen3_8b_enhanced_b.sh

# Print completion information
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
