#!/bin/bash
################################################################################
# Run: qwen3_8b_only_range_trained_unsloth_unquantized_all_test_with_type2
# Fallback for when SLURM is unavailable — run directly in a terminal.
#
# Usage (from repo root):
#   bash scripts/gorina6/run_only_range_unquantized.sh
#   bash scripts/gorina6/run_only_range_unquantized.sh 2>&1 | tee logs/only_range_unquantized.log
################################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="qwen3_8b_only_range_trained_unsloth_unquantized_all_test_with_type2"

echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

bash scripts/inference/slurm_experiment.sh "$EXPERIMENT"

echo ""
echo "=========================================="
echo "Done: $EXPERIMENT"
echo "End Time: $(date)"
echo "=========================================="
