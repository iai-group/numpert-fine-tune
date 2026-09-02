"""Recovers a trainable HuggingFace checkpoint from a GGUF export.

Some fine-tuned checkpoints in this project survive only as GGUF
exports: the merged HuggingFace weights were deleted after conversion,
which leaves the model usable for inference but not for further
training, re-merging at a different rank, or release in trainable form.

``transformers`` can dequantize a GGUF file back into PyTorch tensors
for the architectures listed in
``transformers.integrations.ggml.GGUF_CONFIG_MAPPING`` (Qwen3 among
them), so a checkpoint can be reconstructed from its own export.

Two caveats apply and are reported by this script rather than hidden:

    * Recovery is only as faithful as the export. An ``f16`` GGUF of a
      model merged in ``bfloat16`` has already lost exponent range, so
      the reconstruction is numerically close but not bit-identical to
      the deleted original. Recovering from a quantized export (for
      example ``q4_k_m``) is substantially lossier and is refused unless
      explicitly permitted.
    * Only the *merged* weights can be recovered. The LoRA adapter
      itself is not represented in a merged export and is unrecoverable.

Because the reconstruction is approximate, always verify it by
re-running an evaluation whose published number is known.

Usage:
    python scripts/misc_scripts/recover_hf_from_gguf.py \
        --gguf .models/gguf/<name>/merged_model_r32-f16.gguf \
        --tokenizer unsloth/Qwen3-8B \
        --output .models/<name>/epoch-2/merged_model_r32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

LOSSY_MARKERS = ("q2", "q3", "q4", "q5", "q6", "q8", "iq")


def is_quantized_export(gguf_path: Path) -> bool:
    """Reports whether the export appears to be quantized rather than f16.

    Args:
        gguf_path: Path to the GGUF file.

    Returns:
        True when the filename carries a quantization marker.
    """
    name = gguf_path.name.lower()
    return any(marker in name for marker in LOSSY_MARKERS)


def recover(
    gguf_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    dtype: str = "bfloat16",
) -> None:
    """Dequantizes a GGUF export and saves it as a HuggingFace checkpoint.

    Args:
        gguf_path: The GGUF file to read.
        tokenizer_name: Tokenizer to pair with the recovered weights,
            normally the base model the checkpoint was fine-tuned from.
        output_dir: Directory to write the checkpoint into.
        dtype: Torch dtype to store the recovered weights in.

    Raises:
        FileNotFoundError: If the GGUF file does not exist.
    """
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    logger.info(
        "Reading %s (%.1f GB)", gguf_path, gguf_path.stat().st_size / 2**30
    )
    logger.info("Dequantizing to torch tensors; this is memory-heavy.")
    model = AutoModelForCausalLM.from_pretrained(
        str(gguf_path.parent),
        gguf_file=gguf_path.name,
        dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
    )

    logger.info("Loading tokenizer %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing checkpoint to %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    provenance = {
        "recovered_from_gguf": str(gguf_path),
        "gguf_bytes": gguf_path.stat().st_size,
        "tokenizer": tokenizer_name,
        "stored_dtype": dtype,
        "quantized_source": is_quantized_export(gguf_path),
        "note": (
            "Reconstructed by dequantizing a GGUF export because the "
            "original merged weights were deleted. Numerically close to "
            "but not bit-identical with the original; the LoRA adapter "
            "is not recoverable from a merged export. Verify against a "
            "known published metric before use."
        ),
    }
    with (output_dir / "recovery_provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, indent=2)

    total = sum(p.numel() for p in model.parameters())
    logger.info("Recovered %.2fB parameters", total / 1e9)
    logger.info(
        "Wrote provenance to %s", output_dir / "recovery_provenance.json"
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Recovers a trainable HuggingFace checkpoint from a GGUF "
            "export by dequantizing it."
        )
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="GGUF file to recover from; prefer an f16 export.",
    )
    parser.add_argument(
        "--tokenizer",
        default="unsloth/Qwen3-8B",
        help="Tokenizer to save alongside the weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the recovered checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to store recovered weights in (default: bfloat16).",
    )
    parser.add_argument(
        "--allow-quantized-source",
        action="store_true",
        help=(
            "Permit recovery from a quantized export. Refused by default "
            "because the result is substantially lossier than f16."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Runs the recovery end to end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    if is_quantized_export(args.gguf) and not args.allow_quantized_source:
        raise SystemExit(
            f"{args.gguf.name} looks like a quantized export. Recovery "
            "from it would compound quantization error. Prefer the f16 "
            "export, or pass --allow-quantized-source to override."
        )

    recover(args.gguf, args.tokenizer, args.output, args.dtype)


if __name__ == "__main__":
    main()
