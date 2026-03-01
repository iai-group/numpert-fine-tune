"""Convert a merged HuggingFace model to GGUF quantized format.

Standalone script for converting an already-trained merged model to GGUF.
Also exposes `save_as_gguf()` for use in the fine-tuning pipeline.

Usage:
    python -m src.llm_finetune.utils.training.convert_to_gguf \
        --model_path path/to/merged_model \
        --output_dir path/to/output \
        --quantization_method q4_k_m
"""

import argparse
import os
from typing import Optional

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
    """
    if quantization_method not in SUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"Unsupported quantization method: '{quantization_method}'. "
            f"Supported: {SUPPORTED_QUANT_METHODS}"
        )

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
    """Load a merged model from disk and convert it to GGUF.

    Args:
        model_path: Path to the merged HuggingFace model directory.
        output_dir: Where to write the GGUF output.  Defaults to
            ``<model_path>_gguf_<quantization_method>``.
        quantization_method: Quantization scheme (default: q4_k_m).
        max_seq_length: Maximum sequence length for loading the model.

    Returns:
        The output directory path.
    """
    if output_dir is None:
        output_dir = f"{model_path.rstrip('/')}_gguf_{quantization_method}"

    print(f"Loading model from: {model_path}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    return save_as_gguf(
        model,
        tokenizer,
        output_dir=output_dir,
        quantization_method=quantization_method,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a merged HuggingFace model to GGUF format.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the merged HuggingFace model directory.",
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
