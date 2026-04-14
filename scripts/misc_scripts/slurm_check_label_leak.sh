#!/bin/bash
################################################################################
# SLURM Script: Sanity Check — No Gold Label Leakage in Inference Prompts
#
# Loads test data, applies the chat template, and verifies the gold label
# does not appear in the inference prompt.
#
# USAGE:
#   sbatch scripts/misc_scripts/slurm_check_label_leak.sh
#
#   # Override test file or model:
#   sbatch --export=TEST_JSON=data/perturbed_data/test_split/approximation_True.json \
#          scripts/misc_scripts/slurm_check_label_leak.sh
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=chk-leak
#SBATCH --output=scripts/slurm_out/sanity/check_label_leak_%j.out

# ==============================================================================
# DEFAULTS (override via --export)
# ==============================================================================
TEST_JSON="${TEST_JSON:-data/inference_mini_test.json}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-8B}"
SHOW="${SHOW:-5}"

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p scripts/slurm_out/sanity

echo "=========================================="
echo "SLURM Job Information" a
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Job Name:    ${SLURM_JOB_NAME:-local}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Partition:   ${SLURM_JOB_PARTITION:-local}"
echo "Start Time:  $(date)"
echo ""
echo "TEST_JSON:   $TEST_JSON"
echo "MODEL_NAME:  $MODEL_NAME"
echo "SHOW:        $SHOW"
echo "=========================================="
echo ""

echo "Checking GPU availability..."
nvidia-smi 2>/dev/null || echo "(No GPU detected — running on CPU node?)"
echo ""

# ==============================================================================
# RUN SANITY CHECK
# ==============================================================================
python -m tests.sanity_checks.check_chat_template_no_label_leak \
    --test_json "$TEST_JSON" \
    --model_name "$MODEL_NAME" \
    --show "$SHOW"

EXIT_CODE=$?

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "Exit Code:   $EXIT_CODE"
echo "End Time:    $(date)"
echo "=========================================="

exit $EXIT_CODE
