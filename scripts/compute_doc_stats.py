#!/usr/bin/env python3
"""Compute mean and median word counts for a given key in a JSON/JSONL dataset.

Usage examples:
  python scripts/compute_doc_stats.py data/raw_data/binary_data/test_binary.json
  python scripts/compute_doc_stats.py data/raw_data/binary_data/test_binary.json --key normalized_claim --json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict, Iterable, List, Optional


def get_nested(obj: Dict[str, Any], key_path: str) -> Optional[Any]:
    """Return nested value for a dot-separated key path, or None if missing."""
    if not key_path:
        return None
    cur: Any = obj
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_json(path: str) -> List[Any]:
    """Load either a JSON array/object or a JSONL file. Returns a list of items."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try JSONL (one JSON object per line)
        items: List[Any] = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return items

    if isinstance(data, list):
        return data
    # single object -> return as single-item list
    return [data]


def tokenize_words(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    return text.split()


def compute_lengths(
    items: Iterable[Any], key: str, min_length: int = 0
) -> List[int]:
    lengths: List[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        val = get_nested(it, key)
        if val is None:
            continue
        words = tokenize_words(val)
        if len(words) >= min_length:
            lengths.append(len(words))
    return lengths


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute mean and median word counts for a given key across a JSON dataset"
    )
    p.add_argument("input", help="Input JSON or JSONL file")
    p.add_argument(
        "--key",
        default="doc",
        help="Dot-separated key path to extract (default: doc)",
    )
    p.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Ignore entries shorter than this many words",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output results as JSON",
    )
    args = p.parse_args()

    items = load_json(args.input)
    lengths = compute_lengths(items, args.key, args.min_length)

    if not lengths:
        if args.as_json:
            print(json.dumps({"count": 0, "mean": None, "median": None}))
        else:
            print(
                f'No entries with key "{args.key}" found or no entries passed the filters.'
            )
        return 0

    mean = statistics.mean(lengths)
    median = statistics.median(lengths)
    stdev = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0

    if args.as_json:
        out = {
            "count": len(lengths),
            "mean": mean,
            "median": median,
            "min": min(lengths),
            "max": max(lengths),
            "stdev": stdev,
        }
        print(json.dumps(out))
    else:
        print(f"Count: {len(lengths)}")
        print(f"Mean word count: {mean:.3f}")
        print(f"Median word count: {median:.3f}")
        print(f"Min word count: {min(lengths)}")
        print(f"Max word count: {max(lengths)}")
        print(f"Stddev: {stdev:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
