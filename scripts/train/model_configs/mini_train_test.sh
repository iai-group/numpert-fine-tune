#!/bin/bash

################################################################################
# Model Config: Qwen3-8B — Normal Batching, No type2 from False files
#
# DESCRIPTION:
#   Fine-tunes Qwen3-8B with LoRA on numeration perturbation data using
#   standard random batching, with selective perturbation type loading.
#
#   From *_True.json files: loads type1 (structure-preserving) + type2
#   (value-modifying) perturbations.
#   From *_False.json files: loads type1 only — type2 is excluded because
#   type2 applied to False-labeled claims produces True labels via double
#   negation, which can introduce noisy supervision signal.
#
# BATCHING STRATEGY:
#   - batch_match_claim_id = false → standard random batching
#
# DATASET:
#   Selective perturbation loading via path:type1,type2 syntax.
#   Excludes type2 from False-labeled source files.
#
# BALANCING:
#   Labels balanced (True/False 1:1), perturbed-to-original ratio 1:1,
#   67% T→F / 33% F→F among False perturbations.
#
# USAGE:
#   bash scripts/train/model_configs/qwen3_8b_normal_b_no_type2f.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
MODEL="Qwen3-8B"
RANK=32                         # LoRA rank (16, 32, 64)
START_EPOCH=0                   # 0 for fresh training, 2+ for continued
NUM_TRAIN_EPOCHS=2              # Total number of training epochs
RESUME_MODEL=""                 # Path to merged model when resuming (start_epoch > 0)
PERTURB_TYPE_NAMES="approx_mini_test"

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE=5                    # Per-device batch size
GRAD_ACCUM=1                    # Gradient accumulation steps
                                # Effective batch size = BATCH_SIZE × GRAD_ACCUM = 5
MAX_SEQ_LENGTH=8192             # Maximum sequence length in tokens

# ==============================================================================
# DATASET BALANCING CONFIGURATION
# ==============================================================================
BALANCE_LABELS=true             # Balance True/False label distribution
PERTURB_RATIO=1.0               # Ratio of perturbed to original examples (1.0 = 1:1)
FALSE_T_TO_F_RATIO=0.67         # Among False perturbed examples, % that are T→F
                                # 0.67 = 67% T→F, 33% F→F

# ==============================================================================
# BATCH MATCHING CONFIGURATION (Normal = disabled)
# ==============================================================================
BATCH_MATCH_CLAIM_ID=false      # No claim grouping — standard random batching
SPLIT_LARGE_CLAIMS=false        # N/A when batch_match_claim_id is false
ENSURE_ORIGINAL_IN_BATCH=false  # N/A when batch_match_claim_id is false
ENSURE_PERTURBED_IN_BATCH=false # N/A when batch_match_claim_id is false

# ==============================================================================
# DATA PATHS (with per-file perturbation type filtering)
# ==============================================================================
# Format: path.json:type1,type2  — only load specified perturbation types
# Omit the :suffix to load all types from a file.
#
# From True files:  type1 + type2 (all perturbation types)
# From False files: type1 only    (type2_false excluded)
TRAIN_JSON=(
    "./data/train_mini_test.json:type1,type2"
)

VAL_JSON=(
    "./data/train_mini_test.json:type1,type2"
)

# ==============================================================================
# EVALUATION & CHECKPOINTING
# ==============================================================================
EVAL_STEPS=500                  # Evaluate on validation set every N steps
SAVE_STEPS=500                  # Save checkpoint every N steps

# ==============================================================================
# GGUF QUANTIZATION (optional)
# ==============================================================================
SAVE_GGUF=false                 # Save GGUF-quantized copy after training
GGUF_QUANT_METHOD="q4_k_m"     # Quantization method (q4_k_m, q8_0, q5_k_m, etc.)

# ==============================================================================
# DEBUG
# ==============================================================================
DEBUG=true                     # Print conversation format and tokenisation diagnostics

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

echo "=========================================="
echo "$MODEL Fine-tuning Configuration"
echo "=========================================="
echo ""
echo "Model Configuration:"
echo "  Model:           $MODEL"
echo "  LoRA Rank:       $RANK"
echo "  Start Epoch:     $START_EPOCH"
echo "  Train Epochs:    $NUM_TRAIN_EPOCHS"
if [ -n "$RESUME_MODEL" ]; then
echo "  Resume Model:    $RESUME_MODEL"
fi
echo ""
echo "Training Hyperparameters:"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Grad Accum:      $GRAD_ACCUM"
echo "  Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max Seq Length:  $MAX_SEQ_LENGTH tokens"
echo ""
echo "Dataset Configuration:"
echo "  Balance Labels:  $BALANCE_LABELS"
echo "  Perturb Ratio:   $PERTURB_RATIO"
echo "  False T→F Ratio: $FALSE_T_TO_F_RATIO"
echo ""
echo "Batch Matching (Normal — disabled):"
echo "  Match Claim ID:     $BATCH_MATCH_CLAIM_ID"
echo ""
echo "Selective Perturbation Loading:"
echo "  True files:  type1, type2"
echo "  False files: type1 only (type2 excluded)"
echo ""
echo "Training Data Files:"
for entry in "${TRAIN_JSON[@]}"; do
    file="${entry%%:*}"
    filter="${entry#*:}"
    if [ "$file" = "$filter" ]; then filter="all"; fi
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "  ✓ $file ($size lines) [types: $filter]"
    else
        echo "  ✗ $file (NOT FOUND) [types: $filter]"
    fi
done
echo ""
echo "Validation Data Files:"
for entry in "${VAL_JSON[@]}"; do
    file="${entry%%:*}"
    filter="${entry#*:}"
    if [ "$file" = "$filter" ]; then filter="all"; fi
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "  ✓ $file ($size lines) [types: $filter]"
    else
        echo "  ✗ $file (NOT FOUND) [types: $filter]"
    fi
done
echo ""

# Check if all data files exist (strip :type suffix before checking)
echo "Checking data files..."
all_files_exist=true
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
echo ""

echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

# Build command
CMD="python -m src.llm_finetune.unsloth_finetune \
    --train_json ${TRAIN_JSON[@]} \
    --val_json ${VAL_JSON[@]} \
    --model ${MODEL} \
    --rank $RANK \
    --start_epoch $START_EPOCH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LENGTH \
    --expand_perturbations \
    --perturb_ratio $PERTURB_RATIO \
    --false_perturb_t_to_f_ratio $FALSE_T_TO_F_RATIO \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --perturb_type_names ${PERTURB_TYPE_NAMES}"

# Add optional flags
if [ "$BALANCE_LABELS" = true ]; then
    CMD="$CMD --balance_labels"
fi

if [ "$BATCH_MATCH_CLAIM_ID" = true ]; then
    CMD="$CMD --batch_match_claim_id"
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

# Print full command (for debugging)
echo "Executing command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "Training Completed Successfully!"
echo "=========================================="
echo ""
