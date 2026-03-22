#!/bin/bash
################################################################################
# Inference config: evaluate ONLY normalized_claim across all test split files.
#
# Usage (from repo root):
#   bash scripts/inference/model_configs/inference_all_test_unperturbed.sh
#
# Before running:
#   1) Set MODEL to a local model path or HF model id
#   2) Optionally set OUTPUT_DIR
################################################################################

set -euo pipefail

# Resolve repository root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Model / output configuration
# -----------------------------------------------------------------------------
MODEL="unsloth"
RANK=32
START_EPOCH=0
MAX_SEQ_LENGTH=16384
OUTPUT_DIR="results/inference/unperturbed_all_test"

# -----------------------------------------------------------------------------
# Test files (all files in data/perturbed_data/test_split)
# -----------------------------------------------------------------------------
TEST_JSON=(
  "data/perturbed_data/test_split/approximation_False.json"
  "data/perturbed_data/test_split/approximation_True.json"
  "data/perturbed_data/test_split/negative_False.json"
  "data/perturbed_data/test_split/negative_True.json"
  "data/perturbed_data/test_split/numeration_False.json"
  "data/perturbed_data/test_split/numeration_True.json"
  "data/perturbed_data/test_split/obfuscation_False.json"
  "data/perturbed_data/test_split/obfuscation_True.json"
  "data/perturbed_data/test_split/range_False.json"
  "data/perturbed_data/test_split/range_True.json"
)

echo "=========================================="
echo "Inference: Unperturbed Claims (All Test Files)"
echo "=========================================="
echo "Repo Root:       ${REPO_ROOT}"
echo "Model:           ${MODEL}"
echo "LoRA Rank:       ${RANK}"
echo "Start Epoch:     ${START_EPOCH}"
echo "Max Seq Length:  ${MAX_SEQ_LENGTH}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Test Files:      ${#TEST_JSON[@]}"
echo ""

if [[ "${MODEL}" == "REPLACE_WITH_MODEL_PATH" ]]; then
  echo "ERROR: Please set MODEL in this script before running."
  exit 1
fi

for file in "${TEST_JSON[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: Missing test file: ${file}"
    exit 1
  fi
done

CMD=(
  python -m src.llm_finetune.unsloth_inference
  --test_json
)
CMD+=("${TEST_JSON[@]}")
CMD+=(
  --model "${MODEL}"
  --rank "${RANK}"
  --start_epoch "${START_EPOCH}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --output_dir "${OUTPUT_DIR}"
  --inference_unperturbed_claims
)

echo "Executing command:"
printf ' %q' "${CMD[@]}"
echo

eval "$(printf '%q ' "${CMD[@]}")"

echo ""
echo "=========================================="
echo "Inference Completed Successfully"
echo "=========================================="
