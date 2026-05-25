#!/bin/bash
################################################################################
# Inference Experiment Launcher
#
# Loads an experiment by name from experiments.conf and runs inference.
#
# USAGE:
#   bash scripts/inference/run_experiment.sh <experiment_name>
#   bash scripts/inference/run_experiment.sh --list
#   bash scripts/inference/run_experiment.sh --show <experiment_name>
#   bash scripts/inference/experiments/run_experiment.sh --dry-run <experiment_name>
#
# EXAMPLES:
#   bash scripts/inference/run_experiment.sh qwen3_8b_all_test_filtered
#   bash scripts/inference/run_experiment.sh qwen3_8b_all_test_with_type2
#   bash scripts/inference/run_experiment.sh qwen3_8b_all_test_true_only
#   bash scripts/inference/run_experiment.sh qwen3_32b_all_test_filtered
#
# The "experiment_" prefix is added automatically if omitted.
################################################################################

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/experiments.conf"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==============================================================================
# LOAD EXPERIMENT DEFINITIONS
# ==============================================================================
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: experiments.conf not found at: $CONF_FILE"
    exit 1
fi
source "$CONF_FILE"

cd "${REPO_ROOT}"

# ==============================================================================
# ENVIRONMENT: cuDNN
# Ensure cuDNN from nvidia-cudnn-cu12 package is on LD_LIBRARY_PATH.
# Required for torch to load on nodes where it isn't set automatically.
# ==============================================================================
CUDNN_LIB="$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ ! -d "$CUDNN_LIB" ]; then
    CUDNN_LIB=$(uv run python -c "import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || true)
fi
if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
    echo "Added cuDNN to LD_LIBRARY_PATH: $CUDNN_LIB"
else
    echo "WARNING: cuDNN library directory not found. torch may fail to load."
fi

# ==============================================================================
# LIST ALL AVAILABLE EXPERIMENTS
# ==============================================================================
list_experiments() {
    echo ""
    echo "Available inference experiments:"
    echo "─────────────────────────────────────────────"
    grep -oP '^experiment_\K[a-zA-Z0-9_]+(?=\(\))' "$CONF_FILE" | while read -r name; do
        echo "  $name"
    done
    echo "─────────────────────────────────────────────"
    echo ""
    echo "Usage:  bash $0 <experiment_name>"
    echo "        bash $0 --show <experiment_name>"
    echo "        bash $0 --dry-run <experiment_name>"
    echo ""
}

# ==============================================================================
# SHOW CONFIG FOR AN EXPERIMENT (without running)
# ==============================================================================
show_experiment() {
    local name="$1"
    local func_name="experiment_${name}"

    if ! declare -f "$func_name" > /dev/null 2>&1; then
        echo "ERROR: Unknown experiment: $name"
        echo "Run '$0 --list' to see available experiments."
        exit 1
    fi

    # Call the function to set variables
    "$func_name"

    echo ""
    echo "═══════════════════════════════════════════════"
    echo " Inference Experiment: $name"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "Model Configuration:"
    if [ -n "${LITELLM_MODEL:-}" ]; then
        echo "  litellm model:    $LITELLM_MODEL"
        echo "  Max Workers:      ${MAX_WORKERS:-10}"
        echo "  Max Tokens:       ${MAX_TOKENS:-512}"
        echo "  Temperature:      ${TEMPERATURE:-0.0}"
    else
        echo "  Model:            ${MODEL:-}"
        if [ -n "${TOKENIZER:-}" ]; then
            echo "  Tokenizer:        $TOKENIZER"
        fi
        echo "  LoRA Rank:        ${RANK:-}"
        echo "  Start Epoch:      ${START_EPOCH:-}"
        echo "  Max Seq Length:   ${MAX_SEQ_LENGTH:-} tokens"
        echo ""
        echo "Inference Options:"
        echo "  Use perturbs:   ${USE_PERTURBATIONS:-}"
        echo "  Prompt type:    ${PROMPT_TYPE:-ZERO_SHOT}"
    fi
    if [ -n "$OUTPUT_DIR" ]; then
        echo "  Output Dir:       $OUTPUT_DIR"
    else
        echo "  Output Dir:       (auto-generated)"
    fi
    echo ""
    echo "Test Data:"
    for entry in "${TEST_JSON[@]}"; do
        file="${entry%%:*}"
        filter="${entry#*:}"
        if [ "$file" = "$filter" ]; then filter="all"; fi
        if [ -f "$file" ]; then
            echo "  ✓ $file [types: $filter]"
        else
            echo "  ✗ $file (NOT FOUND) [types: $filter]"
        fi
    done
    echo ""
}

# ==============================================================================
# BUILD AND RUN INFERENCE COMMAND
# ==============================================================================
run_experiment() {
    local name="$1"
    local dry_run="${2:-false}"
    local sample_frac="${3:-}"
    local func_name="experiment_${name}"

    if ! declare -f "$func_name" > /dev/null 2>&1; then
        echo "ERROR: Unknown experiment: $name"
        echo "Run '$0 --list' to see available experiments."
        exit 1
    fi

    # Call the function to set variables
    "$func_name"

    # Show config summary
    show_experiment "$name"

    # Validate data files exist (skip for litellm — Python does glob expansion)
    if [ -z "${LITELLM_MODEL:-}" ]; then
        echo "Checking data files..."
        local all_files_exist=true
        for entry in "${TEST_JSON[@]}"; do
            file="${entry%%:*}"
            if [ ! -f "$file" ]; then
                echo "ERROR: File not found: $file"
                all_files_exist=false
            fi
        done

        if [ "$all_files_exist" = false ]; then
            echo ""
            echo "Please ensure all data files exist before running inference."
            exit 1
        fi
        echo "✓ All data files found"
        echo ""
    fi

    # Build command — litellm models use litellm_inference; PEFT models use
    # peft_inference; all others use unsloth_inference
    if [ -n "${LITELLM_MODEL:-}" ]; then
        CMD="uv run python -m src.llm_finetune.litellm_inference"
        CMD="$CMD --model ${LITELLM_MODEL}"
        CMD="$CMD --test_json \"${TEST_JSON[0]}\""
        CMD="$CMD --max_workers ${MAX_WORKERS:-10}"
        CMD="$CMD --max_tokens ${MAX_TOKENS:-512}"
        CMD="$CMD --temperature ${TEMPERATURE:-0.0}"
        if [ -n "$OUTPUT_DIR" ]; then
            CMD="$CMD --output_dir $OUTPUT_DIR"
        fi
        if [ -n "$sample_frac" ]; then
            CMD="$CMD --sample_frac $sample_frac"
        fi
    elif [ -n "${PEFT_MODEL_PATH:-}" ]; then
        CMD="uv run python -m src.llm_finetune.peft_inference"
        CMD="$CMD --model_path ${PEFT_MODEL_PATH}"
        CMD="$CMD --test_json \"data/perturbed_data/test_split/*.json\""
        CMD="$CMD --max_seq_length $MAX_SEQ_LENGTH"
        CMD="$CMD --batch_size ${BATCH_SIZE:-32}"
        if [ -n "$OUTPUT_DIR" ]; then
            CMD="$CMD --output_dir $OUTPUT_DIR"
        fi
        if [ -n "$sample_frac" ]; then
            CMD="$CMD --sample_frac $sample_frac"
        fi
    else
        CMD="uv run python -m src.llm_finetune.unsloth_inference"
        CMD="$CMD --test_json ${TEST_JSON[*]}"
        CMD="$CMD --model ${MODEL}"
        CMD="$CMD --rank $RANK"
        CMD="$CMD --start_epoch $START_EPOCH"
        CMD="$CMD --max_seq_length $MAX_SEQ_LENGTH"

        if [[ "${USE_PERTURBATIONS,,}" == "true" ]]; then
            CMD="$CMD --use_perturbations"
        else
            CMD="$CMD --no-use_perturbations"
        fi

        if [ -n "$OUTPUT_DIR" ]; then
            CMD="$CMD --output_dir $OUTPUT_DIR"
        fi

        if [ -n "${TOKENIZER:-}" ]; then
            CMD="$CMD --tokenizer $TOKENIZER"
        fi

        CMD="$CMD --prompt_type ${PROMPT_TYPE:-ZERO_SHOT}"

        if [ -n "${EXTRA_INFERENCE_ARGS:-}" ]; then
            CMD="$CMD ${EXTRA_INFERENCE_ARGS}"
        fi
    fi

    echo "═══════════════════════════════════════════════"
    echo " Command"
    echo "═══════════════════════════════════════════════"
    echo "$CMD"
    echo ""

    if [ "$dry_run" = true ]; then
        echo "(dry-run — command not executed)"
        return 0
    fi

    echo "═══════════════════════════════════════════════"
    echo " Starting Inference: $name"
    echo "═══════════════════════════════════════════════"
    echo ""

    eval $CMD

    echo ""
    echo "═══════════════════════════════════════════════"
    echo " Inference Completed: $name"
    echo "═══════════════════════════════════════════════"
}

# ==============================================================================
# MAIN
# ==============================================================================
if [ $# -lt 1 ]; then
    echo "ERROR: No experiment specified."
    list_experiments
    exit 1
fi

case "$1" in
    --list|-l)
        list_experiments
        ;;
    --show|-s)
        if [ $# -lt 2 ]; then
            echo "ERROR: --show requires an experiment name."
            exit 1
        fi
        show_experiment "$2"
        ;;
    --dry-run|-d)
        if [ $# -lt 2 ]; then
            echo "ERROR: --dry-run requires an experiment name."
            exit 1
        fi
        run_experiment "$2" true
        ;;
    --sample|-S)
        if [ $# -lt 3 ]; then
            echo "ERROR: --sample requires <fraction> <experiment_name>."
            echo "Example: $0 --sample 0.05 qwen3_8b_pt_no_numeration"
            exit 1
        fi
        run_experiment "$3" false "$2"
        ;;
    --help|-h)
        echo "Usage: $0 [--list | --show <name> | --dry-run <name> | --sample <frac> <name> | <name>]"
        echo ""
        echo "  <name>               Run inference experiment (experiment_ prefix optional)"
        echo "  --list, -l           List all available experiments"
        echo "  --show, -s           Show config for an experiment without running"
        echo "  --dry-run, -d        Show full command without executing"
        echo "  --sample, -S <frac>  Run on a fraction of test data (e.g. 0.05 for 5%%)"
        echo "  --help, -h           Show this help"
        ;;
    -*)
        echo "ERROR: Unknown option: $1"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
    *)
        run_experiment "$1"
        ;;
esac
