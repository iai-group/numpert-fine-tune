#!/bin/bash
################################################################################
# SLURM Launcher for GGUF Conversion
#
# Converts a merged HuggingFace model to GGUF format with quantization.
# Uses llama.cpp for conversion and supports multiple quantization methods.
#
# USAGE:
#   sbatch scripts/slurm_convert_to_gguf.sh \
#     --model_path <path_to_model> \
#     --output_dir <output_directory> \
#     --quantization_method <method>
#
# EXAMPLES:
#   sbatch scripts/slurm_convert_to_gguf.sh \
#     --model_path ./models/merged_model \
#     --output_dir ./models/merged_model_gguf \
#     --quantization_method q4_k_m
#
#   sbatch scripts/slurm_convert_to_gguf.sh \
#     --model_path /path/to/hf_model \
#     --output_dir /path/to/gguf_output
#
# OVERRIDES (set via environment or sbatch --export):
#   SLURM_PARTITION   (default: gpu)
#   SLURM_TIME        (default: 4:00:00)
#   SLURM_MEM         (default: 300G)
#   SLURM_GPUS        (default: 1)
#   SLURM_NODELIST    (default: unset — let scheduler decide)
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --nodelist=gorina9
#SBATCH --mem=300G
#SBATCH --job-name=gguf-convert
#SBATCH --output=scripts/slurm_out/gguf-convert_%j.out

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================
MODEL_PATH=""
OUTPUT_DIR=""
QUANTIZATION_METHOD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quantization_method)
            QUANTIZATION_METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# VALIDATE ARGUMENTS
# ==============================================================================
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model_path is required"
    echo ""
    echo "Usage: sbatch $0 \\"
    echo "  --model_path <path_to_model> \\"
    echo "  --output_dir <output_directory> \\"
    echo "  [--quantization_method <method>]"
    echo ""
    echo "Supported quantization methods: q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, f16, f32"
    exit 1
fi

# ==============================================================================
# SETUP DIRECTORIES
# ==============================================================================
mkdir -p scripts/slurm_out

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
# Restrict CUDA to GPUs allocated by SLURM
if [ -z "$SLURM_GPUS" ]; then
    # Fallback to SLURM_GPUS_ON_NODE if SLURM_GPUS is not set
    if [ -n "$SLURM_GPUS_ON_NODE" ]; then
        export CUDA_VISIBLE_DEVICES="$SLURM_GPUS_ON_NODE"
        echo "Setting CUDA_VISIBLE_DEVICES to SLURM_GPUS_ON_NODE: $CUDA_VISIBLE_DEVICES"
    fi
else
    export CUDA_VISIBLE_DEVICES="$SLURM_GPUS"
    echo "Setting CUDA_VISIBLE_DEVICES to SLURM_GPUS: $CUDA_VISIBLE_DEVICES"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "================================================================================"
echo "GGUF Conversion Job"
echo "================================================================================"
echo "Job ID             : $SLURM_JOB_ID"
echo "Job Name           : $SLURM_JOB_NAME"
echo "Partition          : $SLURM_PARTITION"
echo "Nodes              : $SLURM_NODELIST"
echo "GPUs               : $SLURM_GPUS_ON_NODE"
echo "================================================================================"
echo "Model Path         : $MODEL_PATH"
echo "Output Directory   : $OUTPUT_DIR"
echo "Quantization       : $QUANTIZATION_METHOD"
echo "================================================================================"

# ==============================================================================
# SETUP & EXECUTE
# ==============================================================================

echo ""
echo "Setting up llama.cpp..."
./scripts/setup_llamacpp.sh

echo ""
echo "Starting GGUF conversion..."
echo "================================================================================"

uv run python -m src.llm_finetune.utils.training.convert_to_gguf \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --quantization_method "$QUANTIZATION_METHOD"

echo "================================================================================"
echo "GGUF conversion completed!"
