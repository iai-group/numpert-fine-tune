#!/bin/bash
################################################################################
# SLURM launcher: run normalized-claims-only inference across all test files.
#
# Submit:
#   sbatch scripts/inference/start_inference_all_test_unperturbed.sh
#
# Edit as needed:
#   - #SBATCH resources (partition, time, GPU count, memory, node)
#   - Python environment activation command
################################################################################
#SBATCH --job-name=qwen-infer-unperturbed-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=scripts/slurm_out/inference/qwen-infer-unperturbed-test.out
#SBATCH --mem=300G
##SBATCH --nodelist=gorina8

set -euo pipefail

mkdir -p scripts/slurm_out/inference

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Job Name:   ${SLURM_JOB_NAME:-N/A}"
echo "Node:       ${SLURM_NODELIST:-N/A}"
echo "Partition:  ${SLURM_JOB_PARTITION:-N/A}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

echo "Activating Python environment..."
source .finetune/bin/activate

echo ""
echo "Checking GPU availability..."
nvidia-smi

echo ""
echo "=========================================="
echo "Starting Inference Script"
echo "=========================================="
echo ""

bash ./scripts/inference/model_configs/inference_all_test_unperturbed.sh

echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
