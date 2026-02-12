#!/bin/bash

################################################################################
# Fine-tuning script for numerical perturbation fact-checking
# Model: Qwen3-8B with LoRA
#
# This script fine-tunes a model to detect numerical perturbations in claims.
# It uses custom batch sampling to group examples from the same claim together.
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
MODEL="Qwen/Qwen3-8B"          # Options: qwen-3-8b, llama-3.2-3b-instruct, llama-3.2-1b-instruct
RANK=32                    # LoRA rank (16, 32, 64) - higher = more parameters but slower
EPOCH=3                    # Epoch number (1 for first training, 2+ for continued training)

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE=6               # Per-device batch size (adjust based on GPU memory)
GRAD_ACCUM=1              # Gradient accumulation steps
                          # Effective batch size = BATCH_SIZE × GRAD_ACCUM = 16
MAX_SEQ_LENGTH=8192      # Maximum sequence length in tokens
                          # Your docs are 2000-4000 words ≈ 3000-6000 tokens

# ==============================================================================
# DATASET BALANCING CONFIGURATION
# ====================  ==========================================================
BALANCE_LABELS=true        # Balance True/False label distribution
PERTURB_RATIO=1.0         # Ratio of perturbed to original examples
                          # 1.0 = 1:1 (equal number of original and perturbed)
                          # 1.5 = 3:2 (1.5× more perturbed than original)
                          
FALSE_T_TO_F_RATIO=0.67   # Among False perturbed examples, what % should be T→F
                          # 0.67 = 67% T→F (True→False), 33% F→F (False→False)
                          # Ensures diversity in how False labels are created

# ==============================================================================
# BATCH MATCHING CONFIGURATION
# ==============================================================================
# These settings control how examples are grouped in batches
BATCH_MATCH_CLAIM_ID=true      # Group examples from same claim in batches
SPLIT_LARGE_CLAIMS=true        # Split claims with many perturbations across batches
ENSURE_ORIGINAL_IN_BATCH=true  # Each batch has ≥1 original example
ENSURE_PERTURBED_IN_BATCH=true # Each batch has ≥1 perturbed example

# ==============================================================================
# DATA PATHS
# ==============================================================================
# Training data - can specify multiple files
TRAIN_JSON=(
    "/home/stud/u2929246/bhome/numpert-fine-tune/data/perturbed_data/train_split/numeration_true_test.json"
)

# Validation data - can specify multiple files
VAL_JSON=(
    "/home/stud/u2929246/bhome/numpert-fine-tune/data/perturbed_data/train_split/numeration_true_test.json"
)

# ==============================================================================
# EVALUATION & CHECKPOINTING
# ==============================================================================
EVAL_STEPS=100            # Evaluate on validation set every N steps
SAVE_STEPS=500            # Save checkpoint every N steps

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

echo "=========================================="
echo "Qwen3-8B Fine-tuning Configuration"
echo "=========================================="
echo ""
echo "Model Configuration:"
echo "  Model:           $MODEL"
echo "  LoRA Rank:       $RANK"
echo "  Epoch:           $EPOCH"
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
echo "Training Data Files:"
for file in "${TRAIN_JSON[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "  ✓ $file ($size lines)"
    else
        echo "  ✗ $file (NOT FOUND)"
    fi
done
echo ""
echo "Validation Data Files:"
for file in "${VAL_JSON[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "  ✓ $file ($size lines)"
    else
        echo "  ✗ $file (NOT FOUND)"
    fi
done
echo ""

# Check if all data files exist
echo "Checking data files..."
all_files_exist=true
for file in "${TRAIN_JSON[@]}" "${VAL_JSON[@]}"; do
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

# Run from the numpert-fine-tune directory (parent of src/)
SCRIPT_DIR="/bhome/numpert-fine-tune"
cd "$SCRIPT_DIR"

CMD="python -m src.llm_finetune.unsloth_finetune \
    --mode finetune \
    --train_json ${TRAIN_JSON[@]} \
    --val_json ${VAL_JSON[@]} \
    --model ${MODEL}"

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
echo "Output Locations:"
echo "  LoRA weights:    epoch-$EPOCH/$MODEL/lora_numpert_r$RANK/"
echo "  Merged model:    epoch-$EPOCH/$MODEL/merged_numpert_r$RANK/"
echo "  Training logs:   epoch-$EPOCH/$MODEL/outputs/numpert_r$RANK/"
echo ""
echo "Next Steps:"
echo "  1. Check WandB dashboard for training metrics"
echo "  2. Run inference with: ./inference_qwen.sh"
echo "  3. For epoch 2, set EPOCH=2 and re-run this script"
echo ""