#!/bin/bash
################################################################################
# SLURM Launcher for Frontier Model Inference (litellm / API-based)
#
# No GPU required — just network access and valid API keys.
# Set ANTHROPIC_API_KEY or OPENAI_API_KEY before submitting.
#
# USAGE:
#   sbatch scripts/inference/slurm_frontier_experiment.sh <experiment_name>
#
# EXAMPLES:
#   sbatch scripts/inference/slurm_frontier_experiment.sh frontier_claude_opus_47
#   sbatch scripts/inference/slurm_frontier_experiment.sh frontier_gpt_55
################################################################################

#SBATCH --partition=cpu
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=numpert_frontier
#SBATCH --output=scripts/slurm_out/peft/frontier_inference_%j.out

# ==============================================================================
# PARSE EXPERIMENT NAME
# ==============================================================================
EXPERIMENT_NAME="${1:-}"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch $0 <experiment_name>"
    exit 1
fi

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "=========================================="
echo "SLURM Frontier Inference"
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
# API KEY CHECKS
# ==============================================================================
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."
    echo "The inference will fail if the experiment requires a missing key."
    echo ""
fi

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
