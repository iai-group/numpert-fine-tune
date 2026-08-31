"""
Estimate total token count for all perturbed numpert data as it would be
sent to an LLM API (e.g. Gemini 2.5).

Each record is formatted as a full inference prompt:
  system prompt + user prompt(claim, doc)

Each file contains records with one claim variant each (original_normalized
or a perturbation). Every record in the split files is counted as one API call.

Tokenizer: tiktoken cl100k_base (GPT-4 / Gemini proxy).
Gemini uses a similar BPE tokenizer; cl100k counts are typically within ~5%.

Usage
-----
    PYTHONPATH=. python scripts/data/estimate_tokens.py
    PYTHONPATH=. python scripts/data/estimate_tokens.py --splits train val test
    PYTHONPATH=. python scripts/data/estimate_tokens.py --data_dir data/spanish/perturbed
"""

import argparse
import json
from pathlib import Path

import tiktoken

from src.numpert_prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_PROMPT,
)

# Gemini pricing
GEMINI_INPUT_PRICE_PER_1M = 1.50  # USD per 1M input tokens
GEMINI_OUTPUT_PRICE_PER_1M = 9.00  # USD per 1M output tokens

# Approximate output: one JSON token per record (~5 tokens: {"label": true/false})
OUTPUT_TOKENS_PER_RECORD = 10


def build_prompt(claim: str, doc: str) -> str:
    user = FACT_CHECK_USER_PROMPT.format(claim=claim, evidence=doc, examples="")
    return FACT_CHECK_SYSTEM_PROMPT + user


def count_tokens_in_file(path: Path, enc: tiktoken.Encoding) -> tuple[int, int]:
    """Return (num_records, total_input_tokens) for all records in a JSON file."""
    records = json.loads(path.read_text())
    total = 0
    for r in records:
        claim = r.get("claim", r.get("normalized_claim", ""))
        doc = r.get("doc", "")
        prompt = build_prompt(claim, doc)
        total += len(enc.encode(prompt))
    return len(records), total


def estimate(data_dir: Path, splits: list[str], enc: tiktoken.Encoding) -> None:
    grand_records = 0
    grand_input_tokens = 0

    for split in splits:
        split_dir = data_dir / f"{split}_split"
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue

        files = sorted(split_dir.glob("*.json"))
        if not files:
            print(f"  [skip] {split_dir} has no JSON files")
            continue

        split_records = 0
        split_tokens = 0
        print(f"\n{'─'*60}")
        print(f"  Split: {split}  ({split_dir})")
        print(f"{'─'*60}")
        print(f"  {'File':<45} {'Records':>8} {'Tokens':>12}")
        print(f"  {'─'*45}  {'─'*8}  {'─'*12}")

        for f in files:
            n, t = count_tokens_in_file(f, enc)
            print(f"  {f.name:<45} {n:>8,} {t:>12,}")
            split_records += n
            split_tokens += t

        output_tokens = split_records * OUTPUT_TOKENS_PER_RECORD
        cost_input = (split_tokens / 1_000_000) * GEMINI_INPUT_PRICE_PER_1M
        cost_output = (output_tokens / 1_000_000) * GEMINI_OUTPUT_PRICE_PER_1M
        cost_total = cost_input + cost_output

        print(f"\n  Total records     : {split_records:>12,}")
        print(
            f"  Total input tokens: {split_tokens:>12,}  ({split_tokens/1e6:.2f}M)"
        )
        print(
            f"  Est. output tokens: {output_tokens:>12,}  ({OUTPUT_TOKENS_PER_RECORD} per record)"
        )
        print(f"  Input cost  ($1.50/1M): ${cost_input:.2f}")
        print(f"  Output cost  ($9.00/1M): ${cost_output:.2f}")
        print(f"  Total cost           : ${cost_total:.2f}")

        grand_records += split_records
        grand_input_tokens += split_tokens

    grand_output = grand_records * OUTPUT_TOKENS_PER_RECORD
    grand_cost_input = (
        grand_input_tokens / 1_000_000
    ) * GEMINI_INPUT_PRICE_PER_1M
    grand_cost_output = (grand_output / 1_000_000) * GEMINI_OUTPUT_PRICE_PER_1M
    grand_cost_total = grand_cost_input + grand_cost_output

    print(f"\n{'═'*60}")
    print(f"  GRAND TOTAL  ({data_dir})")
    print(f"{'═'*60}")
    print(f"  Total records     : {grand_records:>12,}")
    print(
        f"  Total input tokens: {grand_input_tokens:>12,}  ({grand_input_tokens/1e6:.2f}M)"
    )
    print(f"  Est. output tokens: {grand_output:>12,}")
    print(f"  Input cost  ($1.50/1M): ${grand_cost_input:.2f}")
    print(f"  Output cost  ($9.00/1M): ${grand_cost_output:.2f}")
    print(f"  Total cost           : ${grand_cost_total:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/perturbed_data",
        help="Root dir containing <split>_split/ subdirs",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to count (default: train val test)",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Tokenizer: tiktoken cl100k_base (GPT-4/Gemini proxy)")
    print(f"Data dir : {args.data_dir}")
    print(f"Splits   : {args.splits}")

    estimate(Path(args.data_dir), args.splits, enc)


if __name__ == "__main__":
    main()
