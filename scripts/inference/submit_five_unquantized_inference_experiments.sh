#!/bin/bash
################################################################################
# Submit five unquantized inference experiments to SLURM.
#
# Usage:
#   bash scripts/inference/submit_five_unquantized_inference_experiments.sh
################################################################################

set -euo pipefail

EXPERIMENTS=(
    "baseline_unsloth_unquantized_all_test_with_type2"
    "no_approx_trained_unsloth_unquantized_all_test_with_type2"
    "no_num_trained_unsloth_unquantized_all_test_with_type2"
    "no_range_trained_unsloth_unquantized_all_test_with_type2"
    "normalized_claims_trained_unsloth_unquantized_all_test_with_type2"
)

DELAY_SECONDS=5

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available in PATH."
    exit 1
fi

echo "Submitting ${#EXPERIMENTS[@]} unquantized inference experiments..."

for i in "${!EXPERIMENTS[@]}"; do
    experiment_name="${EXPERIMENTS[$i]}"
    echo "[$((i + 1))/${#EXPERIMENTS[@]}] sbatch scripts/inference/slurm_experiment.sh ${experiment_name}"
    sbatch scripts/inference/slurm_experiment.sh "${experiment_name}"

    if [ "$i" -lt $((${#EXPERIMENTS[@]} - 1)) ]; then
        echo "Sleeping ${DELAY_SECONDS}s before next submission..."
        sleep "${DELAY_SECONDS}"
    fi
done

echo "All five unquantized inference experiments submitted."
