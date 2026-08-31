#!/bin/bash
################################################################################
# SLURM Launcher for PEFT Experiments
#
# Submits PEFT experiments defined in scripts/train/experiments.conf via SLURM.
# Automatically sets job name and output path based on experiment name.
#
# USAGE:
#   sbatch scripts/train/slurm_peft_experiment.sh <experiment_name>
#   sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_8b_lora
#
# LIST EXPERIMENTS:
#   bash scripts/train/run_experiment.sh --list | grep peft_
#
# OVERRIDES (set via environment or sbatch --export):
#   SLURM_PARTITION   (default: gpu)
#   SLURM_TIME        (default: 24:00:00)
#   SLURM_MEM         (default: 300G)
#   SLURM_NODELIST    (default: unset — let scheduler decide)
#   SLURM_GPUS        (default: 1)
#   RUN_INFERENCE     (default: false — run inference after training)
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9
#SBATCH --job-name=peft_exp
#SBATCH --output=scripts/slurm_out/peft/peft_experiment_%j.out

# ==============================================================================
# PARSE EXPERIMENT NAME
# ==============================================================================
EXPERIMENT_NAME="${1:-}"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch $0 <experiment_name>"
    echo ""
    echo "Available PEFT experiments:"
    bash scripts/train/run_experiment.sh --list 2>/dev/null | grep "peft_" || echo "  (run: bash scripts/train/run_experiment.sh --list)"
    exit 1
fi

# export CUDA_VISIBLE_DEVICES=5
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ==============================================================================
# SETUP DIRECTORIES
# ==============================================================================
mkdir -p scripts/slurm_out/peft

# ==============================================================================
# JOB INFO
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  PEFT Experiment via SLURM                            ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "SLURM Job Information:"
echo "  Job ID:        ${SLURM_JOB_ID:-local}"
echo "  Job Name:      ${SLURM_JOB_NAME:-local}"
echo "  Node:          ${SLURM_NODELIST:-auto}"
echo "  Partition:     ${SLURM_JOB_PARTITION:-gpu}"
echo "  GPUs:          ${SLURM_GPUS:-1}"
echo "  CPUs per Task: ${SLURM_CPUS_PER_TASK:-8}"
echo "  Memory:        ${SLURM_MEM_PER_NODE:-300G}"
echo "  Time Limit:    ${SLURM_TIME_LIMIT:-24:00:00}"
echo "  Start Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Experiment:    $EXPERIMENT_NAME"
echo "Script:        scripts/train/run_experiment.sh"
echo ""

# ==============================================================================
# ENVIRONMENT
# ==============================================================================
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(No GPU detected)"
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
    echo "Tried: $SCRIPT_DIR/../.. and common locations"
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
# RUN EXPERIMENT
# ==============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                   Running Experiment                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Show experiment config before running
echo "Experiment Configuration:"
bash scripts/train/run_experiment.sh --show "$EXPERIMENT_NAME" 2>/dev/null | head -30
echo ""

# Run the experiment
bash scripts/train/run_experiment.sh "$EXPERIMENT_NAME"
RESULT=$?

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  PEFT Experiment Completed                            ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
if [ $RESULT -eq 0 ]; then
    echo "Status:    SUCCESS ✅"
else
    echo "Status:    FAILED ❌ (exit code: $RESULT)"
fi
echo "End Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results location:"
echo "  Models:  models/"
echo "  Results: results/"
echo "  Logs:    scripts/slurm_out/peft/"
echo ""

exit $RESULT
