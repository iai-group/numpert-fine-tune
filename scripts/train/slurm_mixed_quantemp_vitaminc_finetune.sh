#!/bin/bash
################################################################################
# SLURM Launcher for joint QuanTemp + VitaminC fine-tuning
#
# Trains Qwen3-8B from base on BOTH corpora at once, to test whether one
# model can hold the two numerical-robustness competencies that were shown
# not to transfer to each other:
#   - QuanTemp: synthetic CLAIM-side perturbations (the paper's setting).
#     QuanTemp-only training scores 97.5% here but 78.4% on VitaminC.
#   - VitaminC: naturally occurring EVIDENCE-side numerical revisions.
#     VitaminC-only training scores 98.1% here but 80.9% on QuanTemp.
#
# Data notes:
#   * QuanTemp file list and per-file perturbation-type filters are copied
#     verbatim from experiments.conf::_all_perturb_types_no_f_type2, the
#     recipe that produced the published all-perturb model.
#   * VitaminC records carry no perturb_* fields, so with perturbations
#     enabled they simply contribute their original claim, which is what we
#     want; --use_perturbations stays ON for QuanTemp's sake.
#   * --dedup_mode text_label is required: keying on text alone would drop
#     one snapshot from every VitaminC contrastive pair, while disabling
#     dedup entirely would inflate QuanTemp from ~23.7k to ~36k samples.
#   * The balancer's perturb_ratio / false_perturb_t_to_f_ratio /
#     method_balance_mode are deliberately NOT passed. They are designed
#     around QuanTemp's perturbed-vs-original structure, and would distort
#     the mix by treating 16k VitaminC originals as material to balance
#     perturbations against. Omitting them leaves the balancer neutral.
#
# Sizing: 34,567 training samples; p99 prompt length 3,521 tokens and only
# 0.03% exceed 8,192, so max_seq_length is halved from the published 16,384
# with no practical truncation. Eval runs at quarter-epoch rather than
# every 100 steps, which would otherwise add hours of overhead.
#
# USAGE:
#   sbatch scripts/train/slurm_mixed_quantemp_vitaminc_finetune.sh [epochs]
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mixed_qt_vc_ft
#SBATCH --output=scripts/slurm_out/train/mixed_qt_vc_%j.out

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
START_EPOCH=0
BATCH_SIZE=5
GRAD_ACCUM=1
MAX_SEQ_LENGTH=8192
EVAL_STEPS=0.25
PERTURB_TYPE_NAMES="mixed_quantemp_vitaminc"

QT_TRAIN="data/perturbed_data/train_split"
QT_VAL="data/perturbed_data/val_split"

TRAIN_JSON=(
    "${QT_TRAIN}/numeration_True.json:type1,type2"
    "${QT_TRAIN}/numeration_False.json:type1"
    "${QT_TRAIN}/approximation_True.json:type1,type2"
    "${QT_TRAIN}/approximation_False.json:type1"
    "${QT_TRAIN}/range_True.json:type1,type2"
    "${QT_TRAIN}/range_False.json:type1"
    "${QT_TRAIN}/negative_True.json:type2"
    "${QT_TRAIN}/negative_False.json:"
    "${QT_TRAIN}/obfuscation_True.json:type3_random,type3_masked"
    "${QT_TRAIN}/obfuscation_False.json:type3_random,type3_masked"
    "data/vitaminc/vitaminc_train_sample.json"
)

VAL_JSON=(
    "${QT_VAL}/numeration_True.json:type1,type2"
    "${QT_VAL}/numeration_False.json:type1"
    "${QT_VAL}/approximation_True.json:type1,type2"
    "${QT_VAL}/approximation_False.json:type1"
    "${QT_VAL}/range_True.json:type1,type2"
    "${QT_VAL}/range_False.json:type1"
    "${QT_VAL}/negative_True.json:type2"
    "${QT_VAL}/negative_False.json:"
    "${QT_VAL}/obfuscation_True.json:type3_random,type3_masked"
    "${QT_VAL}/obfuscation_False.json:type3_random,type3_masked"
    "data/vitaminc/vitaminc_val_sample.json"
)

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================
for spec in "${TRAIN_JSON[@]}" "${VAL_JSON[@]}"; do
    path="${spec%%:*}"
    if [ ! -f "$path" ]; then
        echo "ERROR: dataset file not found: $path"
        exit 1
    fi
done

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURM_NODELIST:-local}"
echo "Repo root:    $REPO_ROOT"
echo "Model:        $MODEL (from base)"
echo "Train files:  ${#TRAIN_JSON[@]} (QuanTemp + VitaminC)"
echo "Val files:    ${#VAL_JSON[@]}"
echo "Max seq len:  $MAX_SEQ_LENGTH"
echo "Epochs:       $NUM_TRAIN_EPOCHS"
echo "Start:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "(no GPU detected)"

# ==============================================================================
# TRAIN
# ==============================================================================
CMD=(uv run python -m src.llm_finetune.unsloth_finetune
    --train_json "${TRAIN_JSON[@]}"
    --val_json "${VAL_JSON[@]}"
    --model "$MODEL"
    --rank "$RANK"
    --start_epoch "$START_EPOCH"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum "$GRAD_ACCUM"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --eval_steps "$EVAL_STEPS"
    --use_perturbations
    --dedup_mode text_label
    --perturb_type_names "$PERTURB_TYPE_NAMES")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo "=========================================="
echo "Mixed fine-tuning completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
