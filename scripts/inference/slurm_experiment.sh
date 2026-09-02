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
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gorina9
#SBATCH --job-name=numpert_inference
#SBATCH --output=scripts/slurm_out/peft/peft_inference_%j.out

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
# ENVIRONMENT OVERRIDES
# ==============================================================================
# uv is not on PATH in non-interactive batch environments.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Some GPU nodes here mix A100 and H100 devices. Without PCI ordering, NVML
# and CUDA can disagree on device indices and the job dies at import with
# "Unsloth cannot find any torch accelerator?" — so set it unconditionally.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional GPU pinning: submit with --export=ALL,FORCE_CUDA_DEVICES=<ids>.
if [ -n "${FORCE_CUDA_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$FORCE_CUDA_DEVICES"
    echo "Pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

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
