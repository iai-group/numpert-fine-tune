#!/bin/bash
################################################################################
# SLURM Launcher for PEFT Inference
#
# Runs inference on trained PEFT models with efficiency tracking.
#
# USAGE:
#   # Single model inference
#   sbatch scripts/inference/slurm_peft_inference.sh \
#       --model-path models/checkpoints \
#       --test-data data/test/*.json
#
#   # Batch inference on multiple models
#   sbatch scripts/inference/slurm_peft_inference.sh \
#       --batch-mode \
#       --pattern "*_lora" \
#       --test-data data/test/*.json
#
# ENVIRONMENT VARIABLES (set via sbatch --export):
#   SLURM_PARTITION   (default: gpu)
#   SLURM_TIME        (default: 04:00:00)
#   SLURM_MEM         (default: 100G)
#   SLURM_GPUS        (default: 1)
#   SLURM_NODELIST    (default: unset)
#   SLURM_CPUS_PER_TASK (default: 8)
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gorina9
#SBATCH --job-name=peft_inference
#SBATCH --output=scripts/slurm_out/peft/peft_inference_%j.out

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH="${MODEL_PATH:-}"
TEST_DATA="${TEST_DATA:-data/perturbed_data/test_split/*.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
BATCH_MODE="${BATCH_MODE:-false}"
PATTERN="${PATTERN:-*}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRACK_MEMORY="${TRACK_MEMORY:-true}"

# ==============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-mode)
            BATCH_MODE="true"
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --track-memory)
            TRACK_MEMORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# VALIDATION
# ==============================================================================
mkdir -p scripts/slurm_out/peft

if [ "$BATCH_MODE" = "false" ] && [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model-path is required when not using --batch-mode"
    echo "Usage: sbatch $0 --model-path <path> --test-data <path>"
    exit 1
fi

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                      PEFT Inference Job                               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "SLURM Job Information:"
echo "  Job ID:            ${SLURM_JOB_ID:-local}"
echo "  Job Name:          ${SLURM_JOB_NAME:-local}"
echo "  Node:              ${SLURM_NODELIST:-auto}"
echo "  Partition:         ${SLURM_JOB_PARTITION:-gpu}"
echo "  GPUs:              ${SLURM_GPUS:-1}"
echo "  CPUs per Task:     ${SLURM_CPUS_PER_TASK:-8}"
echo "  Memory:            ${SLURM_MEM_PER_NODE:-100G}"
echo "  Time Limit:        ${SLURM_TIME_LIMIT:-04:00:00}"
echo "  Start Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Configuration:"
if [ "$BATCH_MODE" = "true" ]; then
    echo "  Mode:              Batch Mode (multiple models)"
    echo "  Pattern:           $PATTERN"
else
    echo "  Mode:              Single Model"
    echo "  Model Path:        $MODEL_PATH"
fi
echo "  Test Data:         $TEST_DATA"
echo "  Batch Size:        $BATCH_SIZE"
echo "  Track Memory:      $TRACK_MEMORY"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output Dir:        $OUTPUT_DIR"
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

uv sync --all-extras --quiet

echo "✅ Environment ready"
echo ""

# ==============================================================================
# RUN INFERENCE
# ==============================================================================
if [ "$BATCH_MODE" = "true" ]; then
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║              Running Batch Inference on Multiple Models               ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Find all models matching pattern
    MODELS=$(find models -type d -name "${PATTERN}" 2>/dev/null)

    if [ -z "$MODELS" ]; then
        echo "ERROR: No models found matching pattern: $PATTERN"
        echo "Available models:"
        ls -d models/*/ 2>/dev/null | head -20
        exit 1
    fi

    MODEL_COUNT=$(echo "$MODELS" | wc -l)
    echo "Found $MODEL_COUNT model(s) matching pattern: $PATTERN"
    echo ""

    # Create batch output directory
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="results/batch_inference_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$OUTPUT_DIR"

    COUNTER=0
    for MODEL_DIR in $MODELS; do
        COUNTER=$((COUNTER + 1))
        MODEL_NAME=$(basename "$MODEL_DIR")
        CHECKPOINT_PATH="$MODEL_DIR/checkpoints"

        echo "─────────────────────────────────────────────────────────────────────────"
        echo "[$COUNTER/$MODEL_COUNT] Running inference on: $MODEL_NAME"
        echo "─────────────────────────────────────────────────────────────────────────"

        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "⚠️  Skipping $MODEL_NAME: checkpoints not found at $CHECKPOINT_PATH"
            continue
        fi

        if [ ! -f "$CHECKPOINT_PATH/adapter_config.json" ]; then
            echo "⚠️  Skipping $MODEL_NAME: no adapter_config.json (incomplete training)"
            continue
        fi

        MODEL_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_NAME}"
        mkdir -p "$MODEL_OUTPUT_DIR"

        uv run python -m src.llm_finetune.peft_inference \
            --model_path "$CHECKPOINT_PATH" \
            --test_json "$TEST_DATA" \
            --output_dir "$MODEL_OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --track_memory "$TRACK_MEMORY"

        if [ $? -eq 0 ]; then
            echo "✅ Completed: $MODEL_NAME"
        else
            echo "❌ Failed: $MODEL_NAME"
        fi
        echo ""
    done

    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                  Batch Inference Complete                             ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary of results:"
    for MODEL_DIR in $MODELS; do
        MODEL_NAME=$(basename "$MODEL_DIR")
        METRICS_FILE="$OUTPUT_DIR/${MODEL_NAME}/metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            ACCURACY=$(python -c "import json; m=json.load(open('$METRICS_FILE')); print(f\"{m['accuracy_metrics']['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
            echo "  $MODEL_NAME: Accuracy = $ACCURACY"
        fi
    done

else
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Running Single Model Inference                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="results/inference_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$OUTPUT_DIR"

    echo "Model Path:    $MODEL_PATH"
    echo "Output Dir:    $OUTPUT_DIR"
    echo ""

    uv run python -m src.llm_finetune.peft_inference \
        --model_path "$MODEL_PATH" \
        --test_json "$TEST_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --track_memory "$TRACK_MEMORY"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Inference completed successfully!"
        echo ""
        echo "Results saved to: $OUTPUT_DIR"
        echo "  - metrics.json (accuracy + efficiency metrics)"
        echo "  - predictions.csv (easy to analyze)"
        echo "  - predictions.json (detailed predictions)"
        echo "  - inference_config.json (metadata)"
    else
        echo ""
        echo "❌ Inference failed!"
        exit 1
    fi
fi

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                   PEFT Inference Job Completed                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "End Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Status:    SUCCESS ✅"
echo ""
echo "Next steps:"
echo "  1. Check metrics.json in $OUTPUT_DIR"
echo "  2. Analyze predictions.csv"
echo "  3. Compare results across methods"
echo ""
