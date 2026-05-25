#!/bin/bash
################################################################################
# SLURM: Train all Qwen3-8B balanced prompt-tuning experiments sequentially
#
# Runs 5 leave-one-out PT experiments one after another on a single GPU.
# True class is oversampled to fix low true-recall (class imbalance fix).
#
# USAGE:
#   sbatch scripts/train/slurm_peft_pt_balanced_all.sh
################################################################################

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9
#SBATCH --job-name=peft_pt_balanced_all
#SBATCH --output=scripts/slurm_out/peft/peft_pt_balanced_all_%j.out

EXPERIMENTS=(
    "peft_qwen3_8b_full_data_pt_balanced"
    "peft_qwen3_8b_no_numeration_pt_balanced"
    "peft_qwen3_8b_no_approximation_pt_balanced"
    "peft_qwen3_8b_no_range_pt_balanced"
    "peft_qwen3_8b_no_perturb_split_pt_balanced"
)

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p scripts/slurm_out/peft

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)" || SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    if [ -f "pyproject.toml" ]; then
        REPO_ROOT="$(pwd)"
    else
        for dir in /mnt/beegfs/home/*/repos/numpert-fine-tune /home/*/repos/numpert-fine-tune; do
            if [ -f "$dir/pyproject.toml" ]; then
                REPO_ROOT="$dir"
                break
            fi
        done
    fi
fi

cd "$REPO_ROOT" || { echo "ERROR: Could not find repo root"; exit 1; }
echo "Working directory: $(pwd)"

uv sync --all-extras --quiet
echo "Environment ready"
echo ""

# ==============================================================================
# RUN EXPERIMENTS
# ==============================================================================
FAILED=()
TOTAL=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    echo "════════════════════════════════════════════════════════════════════════"
    echo "[$NUM/$TOTAL] Starting: $EXP"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════════════"

    bash scripts/train/run_experiment.sh "$EXP"
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "✅ [$NUM/$TOTAL] $EXP completed"
    else
        echo "❌ [$NUM/$TOTAL] $EXP FAILED (exit code: $STATUS)"
        FAILED+=("$EXP")
    fi
    echo ""
done

# ==============================================================================
# SUMMARY
# ==============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════════════════"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Completed: $((TOTAL - ${#FAILED[@]}))/$TOTAL"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for exp in "${FAILED[@]}"; do
        echo "  - $exp"
    done
    exit 1
else
    echo "All experiments succeeded ✅"
fi
