#!/bin/bash
################################################################################
# SLURM Launcher for Experiments
#
# Submits any experiment defined in experiments.conf via SLURM.
# Automatically sets job name and output path based on experiment name.
#
# USAGE:
#   sbatch scripts/train/experiments/slurm_experiment.sh <experiment_name>
#
# EXAMPLES:
#   sbatch scripts/train/experiments/slurm_experiment.sh qwen3_8b_numeration_normal
#   sbatch scripts/train/experiments/slurm_experiment.sh qwen3_32b_allperturb_enhanced
#
# OVERRIDES (set via environment or sbatch --export):
#   SLURM_PARTITION   (default: gpu)
#   SLURM_TIME        (default: 24:00:00)
#   SLURM_MEM         (default: 300G)
#   SLURM_NODELIST    (default: unset — let scheduler decide)
#   SLURM_GPUS        (default: 1)
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9
#SBATCH --gres=gpu:nvidia_h100_nvl:1
# These are set dynamically below if possible, defaults here as fallback
#SBATCH --job-name=experiment
#SBATCH --output=scripts/slurm_out/train/finetune_%j.out

# ==============================================================================
# PARSE EXPERIMENT NAME
# ==============================================================================
# When run via sbatch, the experiment name should be passed as argument.
# sbatch passes script arguments after the script path.
EXPERIMENT_NAME="${1:-}"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch $0 <experiment_name>"
    echo ""
    echo "List available experiments:"
    echo "  bash scripts/train/experiments/run_experiment.sh --list"
    exit 1
fi

# ==============================================================================
# SETUP DIRECTORIES
# ==============================================================================
mkdir -p scripts/slurm_out/train

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
#echo "Activating environment..."
#source .finetune/bin/activate

echo ""
echo "Checking GPU availability..."
nvidia-smi 2>/dev/null || echo "(No GPU detected — running on CPU node?)"
echo ""

# ==============================================================================
# RUN EXPERIMENT
# ==============================================================================
bash scripts/train/experiments/run_experiment.sh "$EXPERIMENT_NAME"

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
