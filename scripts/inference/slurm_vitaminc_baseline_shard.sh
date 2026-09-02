#!/bin/bash
################################################################################
# SLURM Launcher for one shard of the VitaminC baseline evaluation
#
# The zero-shot base model emits several hundred tokens of free-form
# reasoning per instance, so a single-process run over the full 11,887
# instance subset takes ~10 hours. Inference is independent per instance,
# so the subset is split into shards (see data/vitaminc/shards/) that run
# concurrently and are merged afterwards by
# scripts/analysis/merge_shard_predictions.py.
#
# The generation flags below MUST stay identical to
# experiment_qwen3_8b_baseline_gguf_vitaminc in experiments.conf, so that
# the merged predictions are equivalent to a single unsharded run.
#
# USAGE:
#   sbatch scripts/inference/slurm_vitaminc_baseline_shard.sh <shard_index>
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=vitaminc_base_shard
#SBATCH --output=scripts/slurm_out/peft/vitaminc_baseline_shard_%j.out

set -euo pipefail

SHARD="${1:-}"
if [ -z "$SHARD" ]; then
    echo "ERROR: no shard index given."
    echo "Usage: sbatch $0 <shard_index>"
    exit 1
fi

# SLURM copies this script into its spool directory, so BASH_SOURCE does
# not point into the repository; use the submission directory instead.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Nodes here mix A100 and H100 devices; without PCI ordering NVML can
# enumerate a device that CUDA cannot then open.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# cuDNN, mirroring scripts/inference/run_experiment.sh.
CUDNN_LIB="$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ ! -d "$CUDNN_LIB" ]; then
    CUDNN_LIB=$(uv run python -c "import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || true)
fi
if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

TEST_JSON="data/vitaminc/shards/vitaminc_test_numerical_shard${SHARD}.json"
OUTPUT_DIR="results/inference/vitaminc/qwen3-8b-q4_k_m/baseline_shard${SHARD}"
MODEL=".models/Qwen3-8B/gguf/Qwen3-8B-baseline_q4_k_m/baseline-q4_k_m.gguf"

if [ ! -f "$TEST_JSON" ]; then
    echo "ERROR: shard file not found: $TEST_JSON"
    exit 1
fi

echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "Shard:    $SHARD"
echo "Data:     $TEST_JSON"
echo "Output:   $OUTPUT_DIR"
echo "Start:    $(date)"
echo "=========================================="

nvidia-smi 2>/dev/null || echo "(no GPU detected)"

uv run python -m src.llm_finetune.unsloth_inference \
    --test_json "$TEST_JSON" \
    --model "$MODEL" \
    --rank 32 \
    --start_epoch 0 \
    --max_seq_length 16384 \
    --no-use_perturbations \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer "unsloth/Qwen3-8B" \
    --prompt_type ZERO_SHOT \
    --no_thinking \
    --dedup_mode none

echo "=========================================="
echo "Shard $SHARD completed at $(date)"
echo "=========================================="
