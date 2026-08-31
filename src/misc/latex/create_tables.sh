#!/bin/bash
################################################################################
# Generate predefined LaTeX tables for qwen3-8b-q4_k_m inference results.
#
# Usage:
#   bash src/latex/create_tables.sh
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -x "${REPO_ROOT}/.numpert-fine-tune/bin/python" ]; then
    DEFAULT_PYTHON_BIN="${REPO_ROOT}/.numpert-fine-tune/bin/python"
else
    DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-qwen3-8b-q4_k_m}"
RESULTS_DIR="results/inference/${RESULTS_SUBDIR}"

cd "${REPO_ROOT}"

run_table() {
    echo "Generating $1"
    shift
    "${PYTHON_BIN}" -m src.latex.generate_accuracy_delta_table "$@"
}

run_table "true_split.tex" \
    --results-root "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/true_split.tex" \
    --model-name-map src/latex/model_name_map.jsonl \
    --baseline-model baseline \
    --source-file-pattern "*_True.json" \
    --perturb-type type1 \
    --class-label true \
    --include-original

run_table "true_split_flipped.tex" \
    --results-root "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/true_split_flipped.tex" \
    --model-name-map src/latex/model_name_map.jsonl \
    --baseline-model baseline \
    --source-file-pattern "*_True.json" \
    --perturb-type type2 \
    --perturb-type type3_masked \
    --perturb-type type3_random \
    --class-label false \
    --include-original
    
run_table "false_split_no_type2.tex" \
    --results-root "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/false_split_no_type2.tex" \
    --model-name-map src/latex/model_name_map.jsonl \
    --baseline-model baseline \
    --source-file-pattern "*_False.json" \
    --perturb-type type1 \
    --perturb-type type3_masked \
    --perturb-type type3_random \
    --group "negative_False.json | perturb_type2" \
    --class-label false \
    --include-original 

echo "Finished generating tables."