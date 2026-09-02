#!/bin/bash
################################################################################
# SLURM launcher for recovering a trainable checkpoint from a GGUF export
#
# Dequantizing an f16 GGUF of an 8B model materialises the weights in
# float32 before casting, so this needs far more RAM than the file size
# suggests. No GPU is required.
#
# USAGE:
#   sbatch scripts/misc_scripts/slurm_recover_gguf.sh <gguf> <output_dir>
################################################################################

#SBATCH --partition=cpu64
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gguf_recover
#SBATCH --output=scripts/slurm_out/gguf_recover_%j.out

set -euo pipefail

GGUF="${1:-}"
OUT="${2:-}"
if [ -z "$GGUF" ] || [ -z "$OUT" ]; then
    echo "Usage: sbatch $0 <gguf_file> <output_dir>"
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p scripts/slurm_out

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH"; exit 1; }

echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-local}"
echo "GGUF:    $GGUF"
echo "Output:  $OUT"
echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

uv run python scripts/misc_scripts/recover_hf_from_gguf.py \
    --gguf "$GGUF" \
    --output "$OUT"

echo "=========================================="
echo "Recovery completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
