#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --job-name=uv_setup
#SBATCH --output=uv_setup.out
#SBATCH --nodelist=gorina8

set -euo pipefail

echo "=========================================="
echo "Environment Setup - Nuclear Rebuild"
echo "=========================================="
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Python: $(command -v python3 || true)"
python3 --version || true
echo ""

# ---- Install/Update uv ----
export UV_INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "${UV_INSTALL_DIR}"

echo "Updating/Installing uv to ${UV_INSTALL_DIR}..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH
export PATH="${HOME}/.local/bin:${PATH}"

echo "uv: $(command -v uv)"
uv --version
echo ""

# ---- NUCLEAR: Remove old environment ----
echo "=========================================="
echo "Removing old environment..."
echo "=========================================="
if [ -d ".numpert_finetune" ]; then
    echo "Nuking .numpert_finetune directory..."
    rm -rf .numpert_finetune
    echo "✓ Old environment removed"
else
    echo "No existing environment found"
fi
echo ""

# Ensure uv is on PATH for this job
export PATH="${HOME}/.local/bin:${PATH}"

echo "uv: $(command -v uv)"
uv --version
uv init --no-workspace
uv python install 3.11
uv venv .finetune --python 3.11 --clear
source .finetune/bin/activate
python -V  # sanity check; should be the .venv python

# 1) Install GPU PyTorch first (CUDA 12.x wheels; cu124 works fine on CUDA 12.5.1)
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio
uv pip install vllm unsloth unsloth_zoo
uv pip install wandb
export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install vllm unsloth unsloth_zoo
uv pip install wandb
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=6
python - <<'PY'
import sys, torch
print("Torch:", torch.__version__, "| CUDA runtime:", torch.version.cuda, "| CUDA available:", torch.cuda.is_available())
for name in ("vllm","unsloth","unsloth_zoo"):
    try:
        __import__(name)
        print(name, "OK")
    except Exception as e:
        print(name, "ERROR:", e, file=sys.stderr)
PY


python - <<'VERIFY'
import sys

def check_import(name, version_attr='__version__'):
    try:
        mod = __import__(name)
        version = getattr(mod, version_attr, 'unknown')
        print(f"✓ {name:20s} {version}")
        return True
    except Exception as e:
        print(f"✗ {name:20s} ERROR: {e}", file=sys.stderr)
        return False

print("\nCore Libraries:")
print("-" * 50)
check_import('numpy')
check_import('scipy')
check_import('sklearn', '__version__')

print("\nPyTorch Ecosystem:")
print("-" * 50)
check_import('torch')
check_import('torchvision')
check_import('torchaudio')

print("\nTransformers Ecosystem:")
print("-" * 50)
check_import('transformers')
check_import('peft')
check_import('accelerate')
check_import('trl')
check_import('datasets')

print("\nUnsloth:")
print("-" * 50)
check_import('unsloth')
check_import('unsloth_zoo')
check_import('vllm')

print("\nOther:")
print("-" * 50)
check_import('wandb')

# CUDA check
print("\nCUDA Status:")
print("-" * 50)
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Test critical import that was failing
print("\nCritical Import Test:")
print("-" * 50)
try:
    from peft.utils.integrations import dequantize_module_weight
    print("✓ peft.utils.integrations import successful")
except Exception as e:
    print(f"✗ peft.utils.integrations FAILED: {e}", file=sys.stderr)

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    print("✓ Unsloth imports successful")
except Exception as e:
    print(f"✗ Unsloth imports FAILED: {e}", file=sys.stderr)

try:
    from sklearn.metrics import f1_score, accuracy_score
    print("✓ sklearn.metrics imports successful")
except Exception as e:
    print(f"✗ sklearn.metrics imports FAILED: {e}", file=sys.stderr)

print("\n" + "=" * 50)
print("Setup Complete!")
print("=" * 50)
VERIFY

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "To activate this environment in future sessions:"
echo "  source .numpert_finetune/bin/activate"
echo ""