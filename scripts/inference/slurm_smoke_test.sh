#!/bin/bash
################################################################################
# Smoke-test SLURM job — single file, all perturbation types
#
# Runs inference on one file (inference_mini_test.json) with all perturbation
# types to verify the full pipeline works before launching a full experiment.
#
# USAGE:
#   sbatch scripts/inference/slurm_smoke_test.sh --model <model>
#   sbatch scripts/inference/slurm_smoke_test.sh --model <model> --tokenizer <hf_id>
#   sbatch scripts/inference/slurm_smoke_test.sh --model <model> --tokenizer <hf_id> --epoch <n>
#
# EXAMPLES (HF/unsloth):
#   sbatch scripts/inference/slurm_smoke_test.sh \
#       --model unsloth/Qwen3-8B
#
# EXAMPLES (GGUF):
#   sbatch scripts/inference/slurm_smoke_test.sh \
#       --model path/to/model.gguf \
#       --tokenizer unsloth/Qwen3-8B
#
# EXAMPLES (fine-tuned checkpoint):
#   sbatch scripts/inference/slurm_smoke_test.sh \
#       --model Qwen3-8B \
#       --epoch 3
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=300G
#SBATCH --job-name=infer-smoke
#SBATCH --output=scripts/slurm_out/inference/smoke_%j.out


# ==============================================================================
# DEFAULTS
# ==============================================================================
MODEL=".models/gguf/Qwen3-8B-daring-ant-evolves_no_approx_no_f2f_type2/merged_model_r32-q4_k_m.gguf"
TOKENIZER="unsloth/Qwen3-8B"
EPOCH=0
RANK=32
MAX_SEQ_LENGTH=16000   # small — smoke test only
TEST_FILE="data/inference_mini_test.json"
OUTPUT_DIR="results/smoke_test"

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";     shift 2 ;;
        --tokenizer)  TOKENIZER="$2"; shift 2 ;;
        --epoch)      EPOCH="$2";     shift 2 ;;
        --rank)       RANK="$2";      shift 2 ;;
        --test_file)  TEST_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2";shift 2 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required."
    echo "Usage: sbatch $0 --model <model_path_or_hf_id> [--tokenizer <hf_id>]"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "=========================================="
echo "SLURM Smoke Test"
echo "=========================================="
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURM_NODELIST:-local}"
echo "Model:          $MODEL"
echo "Epoch:          $EPOCH"
echo "Rank:           $RANK"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Test file:      $TEST_FILE"
echo "Output dir:     $OUTPUT_DIR"
if [ -n "$TOKENIZER" ]; then
    echo "Tokenizer:      $TOKENIZER"
fi
echo "Start time:     $(date)"
echo "=========================================="

nvidia-smi 2>/dev/null || echo "(No GPU detected)"
echo ""

# ==============================================================================
# BUILD AND RUN COMMAND
# ==============================================================================
CMD="uv run python -m src.llm_finetune.unsloth_inference"
CMD="$CMD --test_json ${TEST_FILE}"
CMD="$CMD --model ${MODEL}"
CMD="$CMD --rank ${RANK}"
CMD="$CMD --start_epoch ${EPOCH}"
CMD="$CMD --max_seq_length ${MAX_SEQ_LENGTH}"
CMD="$CMD --use_perturbations"
CMD="$CMD --output_dir ${OUTPUT_DIR}"

if [ -n "$TOKENIZER" ]; then
    CMD="$CMD --tokenizer ${TOKENIZER}"
fi

echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Smoke Test Completed"
echo "End time: $(date)"
echo "=========================================="
