#!/bin/bash
################################################################################
# SLURM launcher for Qwen3-8B with normal (random) batching.
#
# Runs model_configs/qwen3_8b_normal_b.sh which fine-tunes Qwen3-8B using
# standard random batching (no claim grouping). Serves as the baseline
# comparison against the enhanced (claim-matched) batching strategy.
#
# Submit: sbatch scripts/train/start_8b_normal_batch.sh
################################################################################
#SBATCH --job-name=qwen3-8b-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=scripts/slurm_out/train/qwen3-8b-finetune_normal_batch.out
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
source .finetune/bin/activate

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
    
bash ./scripts/train/model_configs/qwen3_8b_normal_b.sh

# Print completion information
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
