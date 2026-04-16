"""Convert a merged HuggingFace model to GGUF quantized format.

Standalone script for converting an already-trained merged model to GGUF.
Also exposes `save_as_gguf()` for use in the fine-tuning pipeline.

Usage:
    # Local model directory
    python -m src.llm_finetune.utils.training.convert_to_gguf \
        --model_path path/to/merged_model \
        --output_dir path/to/output \
        --quantization_method q4_k_m

    # HuggingFace Hub model ID (downloaded automatically)
    python -m src.llm_finetune.utils.training.convert_to_gguf \
        --model_path unsloth/Qwen3-8b \
        --quantization_method q4_k_m
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download

    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False

# Supported GGUF quantization methods (llama.cpp conventions).
SUPPORTED_QUANT_METHODS = [
    "q2_k",
    "q3_k_m",
    "q4_0",
    "q4_k_m",
    "q5_0",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "f32",
]

DEFAULT_QUANT_METHOD = "q4_k_m"


def _build_env_with_cuda_libs() -> dict:
    """Build an env dict with ``LD_LIBRARY_PATH`` pointing to CUDA libs.

    Pip-installed PyTorch bundles CUDA/cuDNN shared libraries under
    ``<site-packages>/nvidia/<pkg>/lib/`` and ``<site-packages>/torch/lib/``.
    These paths aren't automatically on ``LD_LIBRARY_PATH`` for subprocesses,
    causing ``import torch`` to fail with errors like
    ``libcudnn.so.9: cannot open`` or ``libcusparseLt.so.0: cannot open``.

    Uses a subprocess to dynamically discover library paths via Python's
    import system, which is reliable regardless of package manager layout
    (pip, uv, etc.).
    """
    env = os.environ.copy()

    # Ask the target Python to discover nvidia and torch lib directories.
    # This handles uv's hardlink/content-addressable layouts correctly
    # because it resolves paths through Python's own import machinery.
    _DISCOVER_SCRIPT = (
        "import os\n"
        "paths = []\n"
        "try:\n"
        "    import nvidia\n"
        "    base = os.path.dirname(nvidia.__file__)\n"
        "    for d in os.listdir(base):\n"
        "        lib = os.path.join(base, d, 'lib')\n"
        "        if os.path.isdir(lib):\n"
        "            paths.append(lib)\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import torch\n"
        "    paths.append(os.path.join(os.path.dirname(torch.__file__), 'lib'))\n"
        "except Exception:\n"
        "    pass\n"
        "print(os.pathsep.join(paths))\n"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", _DISCOVER_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
        extra = result.stdout.strip()
    except Exception:
        extra = ""

    if extra:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = extra + (
            ":" + existing if existing else ""
        )

    return env


def check_llamacpp_setup() -> Optional[Path]:
    """Check if llama.cpp is properly installed and built.

    Returns:
        Path to llama.cpp root directory if found, None otherwise.

    Raises:
        RuntimeError: If setup is incomplete with instructions.
    """
    # Check if external/llama.cpp exists in project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    llama_cpp_path = project_root / "external" / "llama.cpp"

    if llama_cpp_path.exists():
        converter_script = llama_cpp_path / "convert_hf_to_gguf.py"
        quantizer_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"

        if converter_script.exists() and (
            quantizer_bin.exists() or True
        ):  # converter is main requirement
            return llama_cpp_path

    # If not found, provide setup instructions
    print("\n" + "=" * 80)
    print("⚠️  llama.cpp NOT FOUND - GGUF conversion setup required")
    print("=" * 80)
    print("\nTo enable GGUF export, run the automated setup:")
    print(f"  bash {project_root}/scripts/setup_llamacpp.sh")
    print("\nOr manually:")
    print("  1. Install dependencies: uv add --dev llama-cpp-python")
    print(
        "  2. Clone llama.cpp: "
        "git clone https://github.com/ggerganov/llama.cpp.git "
        "external/llama.cpp"
    )
    print(
        "  3. Build: cmake -S external/llama.cpp -B "
        "external/llama.cpp/build && cmake --build "
        "external/llama.cpp/build -j"
    )
    print("=" * 80 + "\n")

    raise RuntimeError(
        "llama.cpp not found. Run 'bash scripts/setup_llamacpp.sh' "
        "to set up GGUF conversion."
    )


def save_as_gguf(
    model,
    tokenizer,
    output_dir: str,
    quantization_method: str = DEFAULT_QUANT_METHOD,
) -> str:
    """Save a model in GGUF format with the specified quantization.

    Can be called directly after training or from the standalone CLI.

    Args:
        model: The loaded (and optionally merged) FastLanguageModel.
        tokenizer: The associated tokenizer.
        output_dir: Directory where the GGUF file will be written.
        quantization_method: Quantization scheme (default: q4_k_m).

    Returns:
        The output directory path.

    Raises:
        ValueError: If the quantization method is not supported.
        RuntimeError: If llama.cpp is not properly set up.
    """
    if quantization_method not in SUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"Unsupported quantization method: '{quantization_method}'. "
            f"Supported: {SUPPORTED_QUANT_METHODS}"
        )

    # Verify llama.cpp setup before proceeding
    check_llamacpp_setup()

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GGUF QUANTIZATION")
    print("=" * 80)
    print(f"  Output directory : {output_dir}")
    print(f"  Quantization     : {quantization_method}")
    print("=" * 80)

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization_method,
    )

    print(f"GGUF model saved to: {output_dir}")
    return output_dir


def convert_existing_model(
    model_path: str,
    output_dir: Optional[str] = None,
    quantization_method: str = DEFAULT_QUANT_METHOD,
    max_seq_length: int = 16384,
) -> str:
    """Load a merged model from disk (or HuggingFace Hub) and convert it to GGUF.

    Uses llama.cpp's ``convert_hf_to_gguf.py`` and ``llama-quantize``
    directly via subprocess, so **no GPU or CUDA libraries are required**.

    Args:
        model_path: Path to a local merged HuggingFace model directory **or**
            a HuggingFace Hub model ID (e.g. ``unsloth/Qwen2.5-7B-Instruct``).
            If the path does not exist locally it is downloaded via
            ``huggingface_hub.snapshot_download``.
        output_dir: Where to write the GGUF output.  Defaults to
            ``<model_name>_gguf_<quantization_method>`` in the current
            working directory.
        quantization_method: Quantization scheme (default: q4_k_m).
        max_seq_length: Maximum sequence length (unused, kept for API compat).

    Returns:
        The output directory path.

    Raises:
        RuntimeError: If llama.cpp is not properly set up or conversion fails.
    """
    llama_cpp_path = check_llamacpp_setup()

    if quantization_method not in SUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"Unsupported quantization method: '{quantization_method}'. "
            f"Supported: {SUPPORTED_QUANT_METHODS}"
        )

    # If model_path is not a local directory, treat it as a HF Hub model ID.
    if not Path(model_path).is_dir():
        if not _HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is not installed. "
                "Install it with: pip install huggingface-hub"
            )
        print(f"Model path '{model_path}' is not a local directory. "
              "Treating as a HuggingFace Hub model ID and downloading...")
        model_path = snapshot_download(repo_id=model_path)
        print(f"Downloaded to: {model_path}")

    if output_dir is None:
        model_name_for_dir = Path(model_path).name or "model"
        output_dir = f"{model_name_for_dir}_gguf_{quantization_method}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")

    converter_script = llama_cpp_path / "convert_hf_to_gguf.py"
    quantizer_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"

    # Determine output filenames
    model_name = Path(model_path).name or "model"
    f16_gguf = os.path.join(output_dir, f"{model_name}-f16.gguf")
    quantized_gguf = os.path.join(
        output_dir, f"{model_name}-{quantization_method}.gguf"
    )

    print("\n" + "=" * 80)
    print("GGUF CONVERSION")
    print("=" * 80)
    print(f"  Model path       : {model_path}")
    print(f"  Output directory  : {output_dir}")
    print(f"  Quantization      : {quantization_method}")
    print("=" * 80)

    # Build environment with CUDA library paths so torch can load in
    # the subprocess even when LD_LIBRARY_PATH is incomplete.
    env = _build_env_with_cuda_libs()

    # Step 1: Convert HF model to f16 GGUF
    print("\n[1/2] Converting HuggingFace model to f16 GGUF...")
    convert_cmd = [
        sys.executable,
        str(converter_script),
        model_path,
        "--outfile",
        f16_gguf,
        "--outtype",
        "f16",
    ]
    result = subprocess.run(convert_cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed (exit code {result.returncode}). "
            f"Command: {' '.join(convert_cmd)}"
        )

    # Step 2: Quantize (skip if target is already f16 or f32)
    if quantization_method in ("f16", "f32"):
        final_gguf = f16_gguf
        print(f"\nSkipping quantization (method={quantization_method}).")
    else:
        if not quantizer_bin.exists():
            raise RuntimeError(
                f"llama-quantize not found at {quantizer_bin}. "
                "Rebuild llama.cpp with: bash scripts/setup_llamacpp.sh"
            )
        print(f"\n[2/2] Quantizing to {quantization_method}...")
        quant_cmd = [
            str(quantizer_bin),
            f16_gguf,
            quantized_gguf,
            quantization_method,
        ]
        result = subprocess.run(quant_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Quantization failed (exit code {result.returncode}). "
                f"Command: {' '.join(quant_cmd)}"
            )
        final_gguf = quantized_gguf

        # Remove intermediate f16 file to save disk space
        if os.path.exists(f16_gguf) and f16_gguf != final_gguf:
            print(f"Removing intermediate f16 file: {f16_gguf}")
            os.remove(f16_gguf)

    # List output files
    gguf_files = glob.glob(os.path.join(output_dir, "*.gguf"))
    print("\n" + "=" * 80)
    print("GGUF OUTPUT FILES")
    print("=" * 80)
    for f in gguf_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")
    print("=" * 80)

    print(f"\nGGUF model saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a merged HuggingFace model to GGUF format.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the merged HuggingFace model directory, "
            "or a HuggingFace Hub model ID (e.g. unsloth/Qwen2.5-7B-Instruct). "
            "Hub models are downloaded automatically via huggingface_hub."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for the GGUF model. "
            "Defaults to <model_path>_gguf_<quant_method>."
        ),
    )
    parser.add_argument(
        "--quantization_method",
        type=str,
        default=DEFAULT_QUANT_METHOD,
        choices=SUPPORTED_QUANT_METHODS,
        help=f"Quantization method (default: {DEFAULT_QUANT_METHOD}).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Maximum sequence length for loading the model.",
    )

    args = parser.parse_args()

    convert_existing_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization_method=args.quantization_method,
        max_seq_length=args.max_seq_length,
    )
