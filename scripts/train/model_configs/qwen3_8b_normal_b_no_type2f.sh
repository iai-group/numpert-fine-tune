#!/bin/bash

################################################################################
# Fine-tuning script for numerical perturbation fact-checking
# Model: Qwen3-8B with LoRA — Normal batching, no type2_false
#
# This script fine-tunes a model to detect numerical perturbations in claims.
# It uses custom batch sampling to group examples from the same claim together.
#
# Selective loading:
#   - From *_true.json: type1 (structure-preserving) + type2 (value-modifying)
#   - From *_false.json: type1 only (type2_false excluded)
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
MODEL="Qwen3-8B"          # Options: qwen-3-8b, Qwen3-32B, llama-3.2-3b-instruct, llama-3.2-1b-instruct
RANK=32                    # LoRA rank (16, 32, 64) - higher = more parameters but slower
START_EPOCH=1              # Epoch number (1 for first training, 2+ for continued training)
NUM_TRAIN_EPOCHS=2         # Total number of training epochs
PERTURB_TYPE_NAMES="numeration_normal_batched_no_type2f"  # Perturbation type name(s) for logging and saving

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE=5               # Per-device batch size (adjust based on GPU memory)
GRAD_ACCUM=1               # Gradient accumulation steps
                           # Effective batch size = BATCH_SIZE × GRAD_ACCUM = 5
MAX_SEQ_LENGTH=8192        # Maximum sequence length in tokens
                           # Your docs are 2000-4000 words ≈ 3000-6000 tokens

# ==============================================================================
# DATASET BALANCING CONFIGURATION
# ==============================================================================
BALANCE_LABELS=true        # Balance True/False label distribution
PERTURB_RATIO=1.0          # Ratio of perturbed to original examples
                           # 1.0 = 1:1 (equal number of original and perturbed)
FALSE_T_TO_F_RATIO=0.67    # Among False perturbed examples, what % should be T→F
                           # 0.67 = 67% T→F (True→False), 33% F→F (False→False)

# ==============================================================================
# BATCH MATCHING CONFIGURATION
# ==============================================================================
# These settings control how examples are grouped in batches
BATCH_MATCH_CLAIM_ID=false      # Group examples from same claim in batches
SPLIT_LARGE_CLAIMS=false        # Split claims with many perturbations across batches
ENSURE_ORIGINAL_IN_BATCH=false  # Each batch has ≥1 original example
ENSURE_PERTURBED_IN_BATCH=false # Each batch has ≥1 perturbed example

# ==============================================================================
# DATA PATHS (with per-file perturbation type filtering)
# ==============================================================================
# Format: path.json:type1,type2  — only load specified perturbation types
# Omit the :suffix to load all types from a file.
#
# Allowed: type1_true, type2_true, type1_false (+ originals)
# Excluded: type2_false

# Training data
TRAIN_JSON=(
    "./data/perturbed_data/train_split/numeration_true.json:type1,type2"
    "./data/perturbed_data/train_split/numeration_false.json:type1"
)

# Validation data
VAL_JSON=(
    "./data/perturbed_data/val_split/numeration_true.json:type1,type2"
    "./data/perturbed_data/val_split/numeration_false.json:type1"
)

# ==============================================================================
# EVALUATION & CHECKPOINTING
# ==============================================================================
EVAL_STEPS=500             # Evaluate on validation set every N steps
SAVE_STEPS=500             # Save checkpoint every N steps

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
echo "Batch Matching:"
echo "  Match Claim ID:     $BATCH_MATCH_CLAIM_ID"
echo "  Split Large Claims: $SPLIT_LARGE_CLAIMS"
echo "  Ensure Original:    $ENSURE_ORIGINAL_IN_BATCH"
echo "  Ensure Perturbed:   $ENSURE_PERTURBED_IN_BATCH"
echo ""
echo "Selective Perturbation Loading:"
echo "  True file:  type1, type2 (all)"
echo "  False file:  type1 only (type2_false excluded)"
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

# Check if all data files exist
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

CMD="uv run python -m src.llm_finetune.unsloth_finetune\
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

if [ "$SPLIT_LARGE_CLAIMS" = true ]; then
    CMD="$CMD --split_large_claims"
fi

if [ "$ENSURE_ORIGINAL_IN_BATCH" = true ]; then
    CMD="$CMD --ensure_original_in_batch"
fi

if [ "$ENSURE_PERTURBED_IN_BATCH" = true ]; then
    CMD="$CMD --ensure_perturbed_in_batch"
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
