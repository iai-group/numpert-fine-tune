"""Orchestrate vLLM-based number annotation over a dataset split.

Mirrors `run_annotation.py` but uses `VllmAnnotator`, which calls a local
vLLM server instead of the Claude Code CLI. The vLLM server must already be
running before this script is invoked (see `slurm/run_vllm_annotation.sh`).
"""

import argparse

from src.number_annotation.annotators.annotator import (
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    VllmAnnotator,
)
from src.number_annotation.pipeline.run_annotation import annotate_split

INPUT = "data/raw_data/verdict_stripped/binary_rederived/{split}_binary.json"
OUTPUT = "data/spacy_number_annotated/{split}_vllm.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a dataset split with a vLLM-served model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VLLM_MODEL,
        help="Model ID served by vLLM (must match what was passed to vllm serve).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_VLLM_BASE_URL,
        dest="base_url",
        help="Base URL of the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        dest="max_tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    annotator = VllmAnnotator(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    stats = annotate_split(
        INPUT.format(split=args.split),
        OUTPUT.format(split=args.split),
        annotator,
    )
    print(f"Annotated split '{args.split}' (vLLM, {args.model}): {stats}")


if __name__ == "__main__":
    main()
