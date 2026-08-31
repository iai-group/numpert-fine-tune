#!/bin/bash
# Setup llama.cpp for GGUF conversion support
# This script clones and builds llama.cpp with necessary tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="${PROJECT_ROOT}/external/llama.cpp"

echo "=== Setting up llama.cpp for GGUF conversion ==="

# Check for required system dependencies
echo "Checking system dependencies..."
if ! command -v git &> /dev/null; then
    echo "❌ git not found. Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Installing cmake..."
    sudo apt-get install -y cmake
fi

if ! command -v make &> /dev/null; then
    echo "❌ make not found. Installing build-essential..."
    sudo apt-get install -y build-essential
fi

# Clone llama.cpp if not already present
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository..."
    mkdir -p "${PROJECT_ROOT}/external"
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
else
    echo "llama.cpp already cloned. Updating..."
    cd "$LLAMA_CPP_DIR"
    git pull origin master
fi

# Build llama.cpp
echo "Building llama.cpp tools..."
cd "$LLAMA_CPP_DIR"

# Create build directory if it doesn't exist
mkdir -p build

# Configure with CMake (CPU-only build — this is for conversion/quantization
# tools only; GPU inference uses the pre-built llama-cpp-python wheel)
cmake -S . -B build \
    -DGGML_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j "$(nproc)"

echo "✅ llama.cpp successfully built!"
echo "✅ Converter available at: $LLAMA_CPP_DIR/convert_hf_to_gguf.py"
echo "✅ Quantizer available at: $LLAMA_CPP_DIR/build/bin/llama-quantize"

# Try to find and verify Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python >=3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python: $PYTHON_VERSION"

# Install llama-cpp-python with pre-built CUDA wheels (avoids compiling from
# source, which can fail with NVCC/GCC version mismatches on HPC clusters).
# The cu124 index provides wheels compiled against CUDA 12.4.
echo "Installing llama-cpp-python with CUDA support (pre-built wheel)..."
cd "$PROJECT_ROOT"

CUDA_WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu124"

if command -v uv &> /dev/null; then
    uv pip install llama-cpp-python \
        --extra-index-url "$CUDA_WHEEL_INDEX" \
        --force-reinstall --no-cache-dir
elif command -v pip &> /dev/null; then
    pip install llama-cpp-python \
        --extra-index-url "$CUDA_WHEEL_INDEX" \
        --force-reinstall --no-cache-dir
else
    echo "⚠️  Neither uv nor pip found. Install llama-cpp-python manually:"
    echo "   pip install llama-cpp-python --extra-index-url $CUDA_WHEEL_INDEX"
    exit 1
fi

# Verify GPU offload support (use uv run to ensure venv Python is used)
echo "Verifying GPU offload support..."
VERIFY_CMD='
import llama_cpp
if hasattr(llama_cpp, "llama_supports_gpu_offload"):
    assert llama_cpp.llama_supports_gpu_offload(), \
        "llama-cpp-python installed but WITHOUT GPU support!"
    print("✅ llama-cpp-python installed with GPU/CUDA support")
else:
    print("⚠️  Cannot verify GPU support (older API), but wheel should include it")
'
if command -v uv &> /dev/null; then
    uv run python -c "$VERIFY_CMD"
else
    python3 -c "$VERIFY_CMD"
fi

echo ""
echo "=== Setup Complete ==="
echo "To convert a fine-tuned model to GGUF:"
echo "  python external/llama.cpp/convert_hf_to_gguf.py /path/to/model --outfile model.gguf --outtype f16"
echo ""
echo "To quantize the GGUF:"
echo "  external/llama.cpp/build/bin/llama-quantize model.gguf model-q8_0.gguf q8_0"
