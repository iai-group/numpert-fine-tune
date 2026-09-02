#!/bin/bash
################################################################################
# SLURM Launcher for QLoRA fine-tuning on a VitaminC sample
#
# Trains Qwen3-8B on a sample of the VitaminC dataset, to test whether the
# paper's adversarial fine-tuning recipe works on naturally occurring,
# evidence-side numerical contrastive pairs. By default it trains from the
# base model: the merged QuanTemp-tuned checkpoint (young-nightingale) is
# no longer available in trainable form, only as GGUF exports. Set
# RESUME_MODEL to continue from a merged checkpoint once one exists.
#
# VitaminC pairs the SAME claim text with TWO evidence documents (one
# supporting, one refuting); that contrastive structure is the whole
# training signal, so deduplication MUST be keyed on (text, label)
# (--dedup_mode text_label); keying on text alone silently drops one
# member of every pair (16,000 -> 7,987 samples), while disabling
# deduplication entirely would inflate mixed QuanTemp runs.
# VitaminC also carries no perturb_* fields, hence --no-use_perturbations.
#
# USAGE:
#   sbatch scripts/train/slurm_vitaminc_continued_finetune.sh [num_epochs]
#
# Defaults: num_epochs=2
#
# Outputs:
#   Model: .models/Qwen3-8B-<codename>_vitaminc_only/epoch-*/
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=vitaminc_cont_ft
#SBATCH --output=scripts/slurm_out/train/vitaminc_continued_%j.out

set -euo pipefail

NUM_TRAIN_EPOCHS="${1:-2}"

# SLURM copies this script into its spool directory, so BASH_SOURCE does
# not point into the repository; use the submission directory instead.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p scripts/slurm_out/train

# uv is not on PATH in non-interactive batch environments.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH"; exit 1; }

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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL="Qwen3-8B"
RANK=32
# Batch size and accumulation match the published all-perturb 8B recipe
# (scripts/train/experiments.conf::_set_defaults) so that optimisation is
# comparable; only max_seq_length differs, which affects memory and speed
# but not the optimisation itself. VitaminC's longest prompt is 648
# tokens, so 16,384 would waste most of the context window.
BATCH_SIZE=5
GRAD_ACCUM=1
MAX_SEQ_LENGTH=2048
PERTURB_TYPE_NAMES="vitaminc_only"

TRAIN_JSON="data/vitaminc/vitaminc_train_sample.json"
VAL_JSON="data/vitaminc/vitaminc_val_sample.json"

# The merged QuanTemp-tuned 8B checkpoint no longer exists on disk or on
# the Hub (only GGUF exports remain, which cannot be trained from), so by
# default this trains from the base model. Pass RESUME_MODEL to continue
# from a merged checkpoint instead, once one is available again:
#   sbatch --export=ALL,RESUME_MODEL=<path> $0 [num_epochs]
RESUME_MODEL="${RESUME_MODEL:-}"
if [ -n "$RESUME_MODEL" ]; then
    if [ ! -d "$RESUME_MODEL" ]; then
        echo "ERROR: merged resume checkpoint not found: $RESUME_MODEL"
        echo "GGUF exports cannot be loaded for training. Available:"
        find .models -maxdepth 3 -type d -name 'merged_model_r*' \
            2>/dev/null || true
        exit 1
    fi
    START_EPOCH=1        # >0 means "resume": requires --resume_model
    RESUME_ARGS=(--resume_model "$RESUME_MODEL")
    TRAINING_MODE="continued from $RESUME_MODEL"
else
    START_EPOCH=0        # train a fresh adapter on the base model
    RESUME_ARGS=()
    TRAINING_MODE="from base $MODEL"
fi

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================

for f in "$TRAIN_JSON" "$VAL_JSON"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: dataset file not found: $f"
        exit 1
    fi
done

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Job Name:     ${SLURM_JOB_NAME:-local}"
echo "Node:         ${SLURM_NODELIST:-local}"
echo "Repo root:    $REPO_ROOT"
echo "Model:        $MODEL"
echo "Training:     $TRAINING_MODE"
echo "Train data:   $TRAIN_JSON"
echo "Val data:     $VAL_JSON"
echo "Epochs:       $NUM_TRAIN_EPOCHS (from epoch $START_EPOCH)"
echo "Start:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "(no GPU detected)"

# ==============================================================================
# TRAIN
# ==============================================================================
# No explicit `uv sync` here: it prunes ad-hoc packages (llama-cpp-python)
# from the venv; `uv run` performs an inexact sync that keeps them.
CMD=(uv run python -m src.llm_finetune.unsloth_finetune
    --train_json "$TRAIN_JSON"
    --val_json "$VAL_JSON"
    --model "$MODEL"
    --rank "$RANK"
    --start_epoch "$START_EPOCH"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum "$GRAD_ACCUM"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --no-use_perturbations
    --dedup_mode text_label
    --perturb_type_names "$PERTURB_TYPE_NAMES")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo "=========================================="
echo "VitaminC fine-tuning completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
