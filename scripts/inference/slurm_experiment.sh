#!/bin/bash
################################################################################
# SLURM Launcher for Inference Experiments
#
# Submits any inference experiment defined in experiments.conf via SLURM.
# Automatically sets job name and output path based on experiment name.
#
# USAGE:
#   sbatch scripts/inference/slurm_experiment.sh <experiment_name>
#
# EXAMPLES:
#   sbatch scripts/inference/slurm_experiment.sh qwen3_8b_all_test_filtered
#   sbatch scripts/inference/slurm_experiment.sh qwen3_8b_all_test_with_type2
#   sbatch scripts/inference/slurm_experiment.sh qwen3_8b_all_test_true_only
#
# For 32B models, request an H100 GPU:
#   sbatch --gres=gpu:nvidia_h100_nvl:1 \
#          scripts/inference/slurm_experiment.sh qwen3_32b_all_test_filtered
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=300G
# These are set dynamically below if possible, defaults here as fallback
#SBATCH --job-name=numpert_inference
#SBATCH --output=scripts/slurm_out/inference/infer_%j.out

# ==============================================================================
# PARSE EXPERIMENT NAME
# ==============================================================================
EXPERIMENT_NAME="${1:-}"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch $0 <experiment_name>"
    echo ""
    echo "List available experiments:"
    echo "  bash scripts/inference/experiments/run_experiment.sh --list"
    exit 1
fi

# ==============================================================================
# SETUP DIRECTORIES
# ==============================================================================
mkdir -p scripts/slurm_out/inference

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Job Name:    ${SLURM_JOB_NAME:-local}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Partition:   ${SLURM_JOB_PARTITION:-local}"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Start Time:  $(date)"
echo "=========================================="
echo ""

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

echo ""
echo "Checking GPU availability..."
nvidia-smi 2>/dev/null || echo "(No GPU detected — running on CPU node?)"
echo ""

# ==============================================================================
# RUN EXPERIMENT
# ==============================================================================
bash scripts/inference/run_experiment.sh "$EXPERIMENT_NAME"

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "Experiment:  $EXPERIMENT_NAME"
echo "End Time:    $(date)"
echo "=========================================="
