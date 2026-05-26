#!/bin/bash
################################################################################
# Generate predefined LaTeX tables from inference results.
#
# Defaults to scanning every metrics.json under results/inference (one
# combined table per split). Set RESULTS_SUBDIR=<name> to restrict to one
# subtree, e.g.:
#   RESULTS_SUBDIR=q4_k_m bash src/latex/create_tables.sh
#   RESULTS_SUBDIR=unquant MODEL_DIR_MAP=src/latex/model_dir_map_unquant.jsonl \
#       bash src/latex/create_tables.sh
#
# Override the base via RESULTS_BASE (defaults to results/inference).
#
# MODEL_DIR_MAP defaults to src/latex/model_dir_map.jsonl (the q4_k_m map).
# Point it at the unquant map for unquant runs, or set MODEL_DIR_MAP="" to
# disable the dir-map lookup entirely.
#
# BASELINE_MODEL is optional — when set, that folder is pinned to the
# first row. Leave empty (the default) to treat every discovered model
# equally; deltas remain row-self-relative against each row's own Original.
#
# Set PARSE_FAILURES_DISPLAY=column to add a rightmost column showing each
# model's macro-averaged parse-failure rate, or =caption (default) to
# append the per-model breakdown to the caption. Set to an empty string to
# hide parse-failure information entirely.
#
# ABBREVIATE_HEADERS=1 (default) shortens column headers (Approx., Num.,
# Neg., Obfusc., Mask., Rand., Orig.) to keep wide tables compact. Set to
# an empty string to keep full-length headers.
#
# Each split is emitted twice: the base table (e.g. true_split.tex) and a
# *_with_avgs.tex variant that appends a sample-weighted micro-average
# column across non-original cells.
#
# Usage:
#   bash src/latex/create_tables.sh
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -x "${REPO_ROOT}/.numerical-fine-tune/bin/python" ]; then
    DEFAULT_PYTHON_BIN="${REPO_ROOT}/.numerical-fine-tune/bin/python"
else
    DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
RESULTS_BASE="${RESULTS_BASE:-results/inference/unquant}"
RESULTS_SUBDIR="${RESULTS_SUBDIR-}"
if [ -n "${RESULTS_SUBDIR}" ]; then
    RESULTS_DIR="${RESULTS_BASE}/${RESULTS_SUBDIR}"
else
    RESULTS_DIR="${RESULTS_BASE}"
fi
MODEL_NAME_MAP="${MODEL_NAME_MAP:-src/latex/model_name_map.jsonl}"
MODEL_DIR_MAP="${MODEL_DIR_MAP-src/latex/model_dir_map.jsonl}"
PARSE_FAILURES_DISPLAY="${PARSE_FAILURES_DISPLAY-caption}"
ABBREVIATE_HEADERS="${ABBREVIATE_HEADERS-1}"
BASELINE_MODEL="${BASELINE_MODEL-}"
GROUP_SEPARATORS="${GROUP_SEPARATORS-1}"

cd "${REPO_ROOT}"

EXTRA_ARGS=()
if [ -n "${MODEL_DIR_MAP}" ] && [ -f "${MODEL_DIR_MAP}" ]; then
    EXTRA_ARGS+=(--model-dir-map "${MODEL_DIR_MAP}")
fi
if [ -n "${PARSE_FAILURES_DISPLAY}" ]; then
    EXTRA_ARGS+=(--parse-failures-display "${PARSE_FAILURES_DISPLAY}")
fi
if [ -n "${ABBREVIATE_HEADERS}" ] && [ "${ABBREVIATE_HEADERS}" != "0" ]; then
    EXTRA_ARGS+=(--abbreviate-headers)
fi
if [ -n "${BASELINE_MODEL}" ]; then
    EXTRA_ARGS+=(--baseline-model "${BASELINE_MODEL}")
fi
if [ -n "${GROUP_SEPARATORS}" ] && [ "${GROUP_SEPARATORS}" != "0" ]; then
    EXTRA_ARGS+=(--group-separators)
fi

run_table() {
    local basename="$1"
    shift
    echo "Generating ${basename}.tex"
    "${PYTHON_BIN}" -m src.latex.generate_accuracy_delta_table \
        --results-root "${RESULTS_DIR}" \
        --output "${RESULTS_DIR}/${basename}.tex" \
        --model-name-map "${MODEL_NAME_MAP}" \
        "$@" \
        "${EXTRA_ARGS[@]}"
    echo "Generating ${basename}_with_avgs.tex"
    "${PYTHON_BIN}" -m src.latex.generate_accuracy_delta_table \
        --results-root "${RESULTS_DIR}" \
        --output "${RESULTS_DIR}/${basename}_with_avgs.tex" \
        --model-name-map "${MODEL_NAME_MAP}" \
        --micro-average-column \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

run_table true_split \
    --source-file-pattern "*_True.json" \
    --perturb-type type1 \
    --class-label true \
    --include-original

run_table true_split_flipped \
    --source-file-pattern "*_True.json" \
    --perturb-type type2 \
    --perturb-type type3_masked \
    --perturb-type type3_random \
    --class-label false \
    --original-class-label true \
    --include-original

run_table false_split_no_type2 \
    --source-file-pattern "*_False.json" \
    --perturb-type type1 \
    --perturb-type type3_masked \
    --perturb-type type3_random \
    --group "negative_False.json | perturb_type2" \
    --class-label false \
    --include-original

echo "Finished generating tables."