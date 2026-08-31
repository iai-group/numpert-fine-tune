#!/bin/bash
# Quick test/example script for PEFT fine-tuning
# 
# This script demonstrates how to train and evaluate PEFT methods
# on a small subset of data for testing purposes.

set -e

echo "========================================"
echo "PEFT Fine-Tuning Test Script"
echo "========================================"
echo ""

# Configuration
MODEL="Qwen/Qwen2.5-8B-Instruct"
TRAIN_DATA="data/train_mini_test.json"
VAL_DATA="data/train_mini_test.json"  # Use same data for quick test
TEST_DATA="data/train_mini_test.json"

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Please ensure you have test data in the data/ directory"
    exit 1
fi

echo "✅ Found training data: $TRAIN_DATA"
echo ""

# Create output directory
OUTPUT_DIR="./peft_test_results"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo ""

# Test 1: LoRA
echo "========================================"
echo "Test 1: Training LoRA"
echo "========================================"
python -m src.llm_finetune.peft_finetune \
    --train_json "$TRAIN_DATA" \
    --val_json "$VAL_DATA" \
    --model "$MODEL" \
    --peft_method lora \
    --rank 32 \
    --num_train_epochs 1 \
    --batch_size 2 \
    --grad_accum 1 \
    --max_seq_length 2048 \
    --eval_steps 10

echo ""
echo "✅ LoRA training completed"
echo ""

# Test 2: Prompt Tuning
echo "========================================"
echo "Test 2: Training Prompt Tuning"
echo "========================================"
python -m src.llm_finetune.peft_finetune \
    --train_json "$TRAIN_DATA" \
    --val_json "$VAL_DATA" \
    --model "$MODEL" \
    --peft_method prompt_tuning \
    --num_virtual_tokens 20 \
    --num_train_epochs 1 \
    --batch_size 2 \
    --grad_accum 1 \
    --max_seq_length 2048 \
    --eval_steps 10

echo ""
echo "✅ Prompt Tuning training completed"
echo ""

# Test 3: Prefix Tuning
echo "========================================"
echo "Test 3: Training Prefix Tuning"
echo "========================================"
python -m src.llm_finetune.peft_finetune \
    --train_json "$TRAIN_DATA" \
    --val_json "$VAL_DATA" \
    --model "$MODEL" \
    --peft_method prefix_tuning \
    --num_virtual_tokens 20 \
    --num_train_epochs 1 \
    --batch_size 2 \
    --grad_accum 1 \
    --max_seq_length 2048 \
    --eval_steps 10

echo ""
echo "✅ Prefix Tuning training completed"
echo ""

echo "========================================"
echo "All Tests Completed Successfully! ✅"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Check the model checkpoints in models/ directory"
echo "2. Run inference with peft_inference.py"
echo "3. Compare results from different methods"
echo ""
echo "For more details, see PEFT_GUIDE.md"
