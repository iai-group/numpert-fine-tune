#!/bin/bash
################################################################################
# Experiment Launcher
#
# Loads an experiment by name from train/conf and runs training.
#
# USAGE:
#   bash scripts/train/run_experiment.sh <experiment_name>
#   bash scripts/train/run_experiment.sh --list
#   bash scripts/train/run_experiment.sh --show <experiment_name>
#   bash scripts/train/run_experiment.sh --dry-run <experiment_name>
#
# EXAMPLES:
#   bash scripts/train/run_experiment.sh qwen3_8b_numeration_normal
#   bash scripts/train/run_experiment.sh qwen3_32b_allperturb_enhanced
#
# The "experiment_" prefix is added automatically if omitted.
################################################################################

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/experiments.conf"

# ==============================================================================
# LOAD EXPERIMENT DEFINITIONS
# ==============================================================================
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: experiments.conf not found at: $CONF_FILE"
    exit 1
fi
source "$CONF_FILE"

# ==============================================================================
# LIST ALL AVAILABLE EXPERIMENTS
# ==============================================================================
# Resolve the full function name for a given experiment name.
# Tries "experiment_<name>" first, then "exp_<name>".
_resolve_func_name() {
    local name="$1"
    if declare -f "experiment_${name}" > /dev/null 2>&1; then
        echo "experiment_${name}"
    elif declare -f "exp_${name}" > /dev/null 2>&1; then
        echo "exp_${name}"
    else
        echo ""
    fi
}

list_experiments() {
    echo ""
    echo "Available experiments:"
    echo "─────────────────────────────────────────────"
    # Extract all function names starting with experiment_ or exp_
    grep -oP '^(?:experiment_|exp_)\K[a-zA-Z0-9_]+(?=\(\))' "$CONF_FILE" | sort | while read -r name; do
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
    local func_name
    func_name="$(_resolve_func_name "$name")"

    if [ -z "$func_name" ]; then
        echo "ERROR: Unknown experiment: $name"
        echo "Run '$0 --list' to see available experiments."
        exit 1
    fi

    # Call the function to set variables
    "$func_name"

    echo ""
    echo "═══════════════════════════════════════════════"
    echo " Experiment: $name"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "Model Configuration:"
    echo "  Model:            $MODEL"
    echo "  LoRA Rank:        $RANK"
    echo "  Start Epoch:      $START_EPOCH"
    echo "  Train Epochs:     $NUM_TRAIN_EPOCHS"
    echo "  Perturb Type:     $PERTURB_TYPE_NAMES"
    if [ -n "$RESUME_MODEL" ]; then
        echo "  Resume Model:     $RESUME_MODEL"
    fi
    echo ""
    echo "Training Hyperparameters:"
    echo "  Batch Size:       $BATCH_SIZE"
    echo "  Grad Accum:       $GRAD_ACCUM"
    echo "  Effective Batch:  $((BATCH_SIZE * GRAD_ACCUM))"
    echo "  Max Seq Length:   $MAX_SEQ_LENGTH tokens"
    echo ""
    echo "Dataset Balancing:"
    echo "  Method:           $METHOD_BALANCE_MODE"
    echo "  Perturb Ratio:    $PERTURB_RATIO"
    echo "  False T→F Ratio:  $FALSE_PERTURB_T_TO_F_RATIO"
    echo ""
    echo "Batch Matching:"
    echo "  Match Claim ID:   $BATCH_MATCH_CLAIM_ID"
    if [ "$BATCH_MATCH_CLAIM_ID" = true ]; then
        echo "  Split Large:      $SPLIT_LARGE_CLAIMS"
        echo "  Ensure Original:  $ENSURE_ORIGINAL_IN_BATCH"
        echo "  Ensure Perturbed: $ENSURE_PERTURBED_IN_BATCH"
    fi
    echo ""
    echo "Eval & Checkpointing:"
    echo "  Eval Steps:       $EVAL_STEPS"
    echo "  Save Steps:       After every eval"
    echo ""
    echo "Training Data:"
    for entry in "${TRAIN_JSON[@]}"; do
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
    echo "Validation Data:"
    for entry in "${VAL_JSON[@]}"; do
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
# BUILD AND RUN TRAINING COMMAND
# ==============================================================================
run_experiment() {
    local name="$1"
    local dry_run="${2:-false}"
    local func_name
    func_name="$(_resolve_func_name "$name")"

    if [ -z "$func_name" ]; then
        echo "ERROR: Unknown experiment: $name"
        echo "Run '$0 --list' to see available experiments."
        exit 1
    fi

    # Call the function to set variables
    "$func_name"

    # Show config summary
    show_experiment "$name"

    # Validate data files exist (skipped when no training epochs / no data files)
    if [ "${#TRAIN_JSON[@]}" -eq 0 ]; then
        echo "No data files configured (num_train_epochs=0): skipping data file validation"
    else
        echo "Checking data files..."
        local all_files_exist=true
        for entry in "${TRAIN_JSON[@]}" "${VAL_JSON[@]}"; do
            file="${entry%%:*}"
            if [ ! -f "$file" ]; then
                echo "ERROR: File not found: $file"
                all_files_exist=false
            fi
        done

        if [ "$all_files_exist" = false ]; then
            echo ""
            echo "Please ensure all data files exist before running training."
            exit 1
        fi
        echo "✓ All data files found"
    fi
    echo ""

    # Build command
    CMD="uv run python -m src.llm_finetune.unsloth_finetune \
        --model ${MODEL} \
        --rank $RANK \
        --start_epoch $START_EPOCH \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_seq_length $MAX_SEQ_LENGTH \
        --method_balance_mode $METHOD_BALANCE_MODE \
        --perturb_ratio $PERTURB_RATIO \
        --false_perturb_t_to_f_ratio $FALSE_PERTURB_T_TO_F_RATIO \
        --eval_steps $EVAL_STEPS \
        --perturb_type_names ${PERTURB_TYPE_NAMES}"

    # Add data args only when data files are configured
    if [ "${#TRAIN_JSON[@]}" -gt 0 ]; then
        CMD="$CMD --train_json ${TRAIN_JSON[@]} --val_json ${VAL_JSON[@]}"
    fi

    # Add optional flags
    if [[ "${USE_PERTURBATIONS,,}" == "true" ]]; then
        CMD="$CMD --use_perturbations"
    else
        CMD="$CMD --no-use_perturbations"
    fi

    if [ "$BATCH_MATCH_CLAIM_ID" = true ]; then
        CMD="$CMD --batch_match_claim_id"
    fi

    if [ "$SPLIT_LARGE_CLAIMS" = true ]; then
        CMD="$CMD --split_large_claims"
    fi

    if [ "$ENSURE_ORIGINAL_IN_BATCH" = true ]; then
        CMD="$CMD --ensure_original_in_batch"
    fi

    if [ "$ENSURE_PERTURBED_IN_BATCH" = true ]; then
        CMD="$CMD --ensure_perturbed_in_batch"
    fi

    if [ -n "$RESUME_MODEL" ]; then
        CMD="$CMD --resume_model $RESUME_MODEL"
    fi

    if [ "$SAVE_GGUF" = true ]; then
        CMD="$CMD --save_gguf --gguf_quantization_method $GGUF_QUANT_METHOD"
    fi

    if [ "$DEBUG" = true ]; then
        CMD="$CMD --debug"
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
    echo " Starting Training: $name"
    echo "═══════════════════════════════════════════════"
    echo ""

    eval $CMD

    echo ""
    echo "═══════════════════════════════════════════════"
    echo " Training Completed: $name"
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
    --help|-h)
        echo "Usage: $0 [--list | --show <name> | --dry-run <name> | <name>]"
        echo ""
        echo "  <name>          Run experiment (experiment_ prefix optional)"
        echo "  --list, -l      List all available experiments"
        echo "  --show, -s      Show config for an experiment without running"
        echo "  --dry-run, -d   Show full command without executing"
        echo "  --help, -h      Show this help"
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
