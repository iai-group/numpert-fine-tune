"""Smoke-run the vLLM annotation track on four examples.

Reuses the shared smoke-input file produced by `run_smoke` so the sample is
identical across tracks (enabling apples-to-apples comparison).  If the
persisted input is absent it is built first.

The vLLM server must already be running before this script is invoked::

    vllm serve "Qwen/Qwen3.6-35B-A3B" --port 8000

Then run::

    python -m src.number_annotation.smoke.run_smoke_vllm
"""

import argparse
import os

from src.number_annotation.annotators.annotator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    VllmAnnotator,
)
from src.number_annotation.pipeline.run_annotation import annotate_split
from src.number_annotation.smoke.run_smoke import (
    INPUT,
    SMOKE_INPUT,
    build_smoke_input,
)

SMOKE_OUTPUT = "data/spacy_number_annotated/smoke_vllm_four_samples.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-run the vLLM annotation track (4 examples).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        default=DEFAULT_VLLM_MODEL,
        help="Model ID served by vLLM.",
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
        default=DEFAULT_MAX_TOKENS,
        dest="max_tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    if not os.path.exists(SMOKE_INPUT):
        count = build_smoke_input(
            INPUT.format(split=args.split), SMOKE_INPUT, args.n, args.seed
        )
        print(f"Built smoke input: {count} records -> {SMOKE_INPUT}")

    annotator = VllmAnnotator(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    stats = annotate_split(SMOKE_INPUT, SMOKE_OUTPUT, annotator)
    print(f"vLLM smoke ({args.model}) -> {SMOKE_OUTPUT}")
    print(f"Stats: {stats}")
    print(
        "Compare with: python -m src.number_annotation.eval.compare_models "
        "--models opus vllm --ref opus --html compare_vllm.html"
    )


if __name__ == "__main__":
    main()
