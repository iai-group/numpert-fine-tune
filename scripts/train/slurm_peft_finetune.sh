#!/bin/bash
################################################################################
# SLURM Launcher for PEFT Fine-Tuning
#
# Submits PEFT fine-tuning jobs via SLURM with support for:
# - Single method training (LoRA, Prompt Tuning, Prefix Tuning, etc.)
# - Multi-method comparison runs
# - Inference on trained models
#
# USAGE:
#   # Train single method
#   sbatch scripts/train/slurm_peft_finetune.sh \
#       --method lora \
#       --epochs 3 \
#       --batch-size 2
#
#   # Train and convert to GGUF
#   sbatch scripts/train/slurm_peft_finetune.sh \
#       --method lora \
#       --epochs 3 \
#       --batch-size 2 \
#       --save-as-gguf \
#       --gguf-quantization q4_k_m
#
#   # Train all methods for comparison
#   sbatch scripts/train/slurm_peft_finetune.sh \
#       --compare \
#       --epochs 1 \
#       --batch-size 2
#
# ENVIRONMENT VARIABLES (set via sbatch --export):
#   SLURM_PARTITION   (default: gpu)
#   SLURM_TIME        (default: 24:00:00)
#   SLURM_MEM         (default: 300G)
#   SLURM_GPUS        (default: 1)
#   SLURM_NODELIST    (default: gorina9)
#   SLURM_CPUS_PER_TASK (default: 8)
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9
#SBATCH --cpus-per-task=8
#SBATCH --job-name=peft_finetune
#SBATCH --output=scripts/slurm_out/peft/peft_finetune_8b_r32.out

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default values (can be overridden via command-line args or SLURM --export)
PEFT_METHOD="${PEFT_METHOD:-lora}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RANK="${RANK:-32}"
NUM_VIRTUAL_TOKENS="${NUM_VIRTUAL_TOKENS:-20}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
COMPARE_MODE="${COMPARE_MODE:-false}"
RUN_INFERENCE="${RUN_INFERENCE:-false}"
SAMPLE_PERCENT="${SAMPLE_PERCENT:-100}"
SAVE_AS_GGUF="${SAVE_AS_GGUF:-false}"
GGUF_QUANTIZATION="${GGUF_QUANTIZATION:-q4_k_m}"
# SAMPLE_PERCENT=10
CUDA_VISIBLE_DEVICES=7
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_DEVICE_ORDER=PCI_BUS_ID

# Data paths
TRAIN_DATA="${TRAIN_DATA:-data/perturbed_data/train_split/*.json}"
VAL_DATA="${VAL_DATA:-data/perturbed_data/val_split/*.json}"
TEST_DATA="${TEST_DATA:-data/perturbed_data/test_split/*.json}"

# ==============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            PEFT_METHOD="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --virtual-tokens)
            NUM_VIRTUAL_TOKENS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --compare)
            COMPARE_MODE="true"
            shift
            ;;
        --with-inference)
            RUN_INFERENCE="true"
            shift
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --val-data)
            VAL_DATA="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --sample-percent)
            SAMPLE_PERCENT="$2"
            shift 2
            ;;
        --save-as-gguf)
            SAVE_AS_GGUF="true"
            shift
            ;;
        --gguf-quantization)
            GGUF_QUANTIZATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p scripts/slurm_out/peft

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    PEFT Fine-Tuning Job                               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "SLURM Job Information:"
echo "  Job ID:            ${SLURM_JOB_ID:-local}"
echo "  Job Name:          ${SLURM_JOB_NAME:-local}"
echo "  Node:              ${SLURM_NODELIST:-local}"
echo "  Partition:         ${SLURM_JOB_PARTITION:-gpu}"
echo "  GPUs:              ${SLURM_GPUS:-1}"
echo "  CPUs per Task:     ${SLURM_CPUS_PER_TASK:-8}"
echo "  Memory:            ${SLURM_MEM_PER_NODE:-300G}"
echo "  Time Limit:        ${SLURM_TIME_LIMIT:-24:00:00}"
echo "  Start Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Configuration:"
if [ "$COMPARE_MODE" = "true" ]; then
    echo "  Mode:              COMPARISON (all methods)"
else
    echo "  Mode:              Single Method"
    echo "  PEFT Method:       $PEFT_METHOD"
fi
echo "  Model:             $MODEL"
echo "  Training Epochs:   $NUM_EPOCHS"
echo "  Batch Size:        $BATCH_SIZE"
echo "  Gradient Accum:    $GRAD_ACCUM"
if [ "$PEFT_METHOD" = "lora" ] || [ "$PEFT_METHOD" = "lora_prompt" ] || [ "$PEFT_METHOD" = "lora_prefix" ]; then
    echo "  LoRA Rank:         $RANK"
fi
if [ "$PEFT_METHOD" != "lora" ]; then
    echo "  Virtual Tokens:    $NUM_VIRTUAL_TOKENS"
fi
echo "  Max Seq Length:    $MAX_SEQ_LENGTH"
echo "  Sample Percent:    $SAMPLE_PERCENT%"
echo "  Train Data:        $TRAIN_DATA"
echo "  Val Data:          $VAL_DATA"
if [ "$RUN_INFERENCE" = "true" ]; then
    echo "  Test Data:         $TEST_DATA"
    echo "  Run Inference:     YES"
fi
if [ "$SAVE_AS_GGUF" = "true" ]; then
    echo "  Save as GGUF:      YES (quantization: $GGUF_QUANTIZATION)"
fi
echo ""

# ==============================================================================
# GPU CHECK
# ==============================================================================
echo "GPU Availability:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (No GPU detected)"
echo ""

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
echo "Setting up Python environment with uv..."

# Change to repo root directory
# Use multiple methods to detect repo root for compatibility with SLURM
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)" || SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

# If we're in /var/spool or similar, try to find repo root from current working directory
if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    # Try the current working directory
    if [ -f "pyproject.toml" ]; then
        REPO_ROOT="$(pwd)"
    else
        # Try common locations
        for dir in /mnt/beegfs/home/*/repos/numpert-fine-tune /home/*/repos/numpert-fine-tune /opt/*/numpert-fine-tune; do
            if [ -f "$dir/pyproject.toml" ]; then
                REPO_ROOT="$dir"
                break
            fi
        done
    fi
fi

cd "$REPO_ROOT" || {
    echo "ERROR: Failed to find repo root with pyproject.toml"
    exit 1
}

echo "Working directory: $(pwd)"
echo ""

# Verify we're in the right place
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in $(pwd)"
    exit 1
fi

# Sync dependencies
uv sync --all-extras --quiet

echo "✅ Environment ready"
echo ""

# ==============================================================================
# RUN TRAINING
# ==============================================================================
if [ "$COMPARE_MODE" = "true" ]; then
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                Running PEFT Method Comparison                         ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    METHODS="lora,prompt_tuning,prefix_tuning,lora_prompt,lora_prefix"

    uv run python -m src.llm_finetune.compare_peft_methods \
        --train_json "$TRAIN_DATA" \
        --val_json "$VAL_DATA" \
        --test_json "$TEST_DATA" \
        --methods "$METHODS" \
        --model "$MODEL" \
        --rank "$RANK" \
        --num_virtual_tokens "$NUM_VIRTUAL_TOKENS" \
        --num_train_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --sample_percent "$SAMPLE_PERCENT" \
        --output_dir "comparison_results"

    echo ""
    echo "✅ Comparison complete!"
    echo ""
    echo "Results saved to:"
    ls -d comparison_results/comparison_* | tail -1 | while read dir; do
        echo "  $dir"
        echo "  - comparison_report.csv (side-by-side comparison)"
        echo "  - [method]/ (individual results)"
    done

else
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                  Running PEFT Fine-Tuning                             ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Method: $PEFT_METHOD"
    echo ""

    # Build command
    CMD="CUDA_VISIBLE_DEVICES=7 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run python -m src.llm_finetune.peft_finetune"
    CMD="$CMD --train_json '$TRAIN_DATA'"
    CMD="$CMD --val_json '$VAL_DATA'"
    CMD="$CMD --model '$MODEL'"
    CMD="$CMD --peft_method '$PEFT_METHOD'"
    CMD="$CMD --rank '$RANK'"
    CMD="$CMD --num_virtual_tokens '$NUM_VIRTUAL_TOKENS'"
    CMD="$CMD --num_train_epochs '$NUM_EPOCHS'"
    CMD="$CMD --batch_size '$BATCH_SIZE'"
    CMD="$CMD --grad_accum '$GRAD_ACCUM'"
    CMD="$CMD --max_seq_length '$MAX_SEQ_LENGTH'"
    CMD="$CMD --sample_percent '$SAMPLE_PERCENT'"

    # Run training
    eval "$CMD"

    echo ""
    echo "✅ Training complete!"
    echo ""

    # ==============================================================================
    # CONVERT TO GGUF (OPTIONAL)
    # ==============================================================================
    if [ "$SAVE_AS_GGUF" = "true" ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║                  Converting Model to GGUF Format                       ║"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""

        # Find the latest model checkpoint
        LATEST_MODEL=$(find models -name "*_${PEFT_METHOD}" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -z "$LATEST_MODEL" ]; then
            echo "WARNING: Could not find trained model for $PEFT_METHOD"
            echo "Skipping GGUF conversion"
        else
            MODEL_PATH="$LATEST_MODEL"
            GGUF_OUTPUT_DIR="${LATEST_MODEL}_gguf_${GGUF_QUANTIZATION}"

            echo "Source model: $MODEL_PATH"
            echo "Output directory: $GGUF_OUTPUT_DIR"
            echo "Quantization: $GGUF_QUANTIZATION"
            echo ""

            # Run GGUF conversion
            uv run python external/llama.cpp/convert_hf_to_gguf.py \
                "$MODEL_PATH" \
                --outtype f16 \
                --outfile "$GGUF_OUTPUT_DIR/model-fp16.gguf"

            if [ $? -eq 0 ]; then
                echo ""
                echo "Running quantization with $GGUF_QUANTIZATION..."
                
                # Run quantization
                external/llama.cpp/quantize "$GGUF_OUTPUT_DIR/model-fp16.gguf" \
                    "$GGUF_OUTPUT_DIR/model-${GGUF_QUANTIZATION}.gguf" \
                    "$GGUF_QUANTIZATION"

                if [ $? -eq 0 ]; then
                    echo ""
                    echo "✅ GGUF conversion complete!"
                    echo ""
                    echo "GGUF models saved to: $GGUF_OUTPUT_DIR"
                    echo "  - model-fp16.gguf (full precision)"
                    echo "  - model-${GGUF_QUANTIZATION}.gguf (quantized)"
                else
                    echo ""
                    echo "⚠️  Quantization failed, but fp16 model is available"
                fi
            else
                echo ""
                echo "ERROR: GGUF conversion failed"
                exit 1
            fi
        fi

        echo ""
    fi

    # ==============================================================================
    # RUN INFERENCE (OPTIONAL)
    # ==============================================================================
    if [ "$RUN_INFERENCE" = "true" ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║                      Running Inference                                ║"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""

        # Find the latest model checkpoint
        LATEST_MODEL=$(find models -name "*_${PEFT_METHOD}" -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -z "$LATEST_MODEL" ]; then
            echo "ERROR: Could not find trained model for $PEFT_METHOD"
            exit 1
        fi

        MODEL_PATH="$LATEST_MODEL/checkpoints"
        OUTPUT_DIR="results/${PEFT_METHOD}_inference_$(date +%Y%m%d_%H%M%S)"

        uv run python -m src.llm_finetune.peft_inference \
            --model_path "$MODEL_PATH" \
            --test_json "$TEST_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --track_memory True

        echo ""
        echo "✅ Inference complete!"
        echo ""
        echo "Results saved to: $OUTPUT_DIR"
        echo "  - metrics.json (accuracy + efficiency)"
        echo "  - predictions.csv (easy analysis)"
        echo "  - predictions.json (detailed)"
    fi
fi

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    PEFT Job Completed                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "End Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Status:    SUCCESS ✅"
echo ""
echo "Next steps:"
echo "  1. Check results in models/ or comparison_results/"
echo "  2. View metrics.json in results/"
echo "  3. Compare with comparison_report.csv"
echo ""
