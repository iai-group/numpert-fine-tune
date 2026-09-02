"""Merges per-shard inference predictions into one run directory.

Long baseline evaluations are split across shards that run concurrently
(see ``scripts/inference/slurm_vitaminc_baseline_shard.sh``).  Because
inference is independent per instance, concatenating the shards'
``predictions.csv`` files reproduces exactly what a single unsharded run
would have written, provided the shards partition the evaluation set
without overlap.  This module performs that concatenation and verifies
the partition property before writing anything.

Usage:
    python scripts/analysis/merge_shard_predictions.py \
        --shard-dirs \
            results/inference/vitaminc/qwen3-8b-q4_k_m/baseline_shard* \
        --output \
            results/inference/vitaminc/qwen3-8b-q4_k_m/baseline_merged \
        --expected data/vitaminc/vitaminc_test_numerical.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_predictions_csv(run_dir: Path) -> Optional[Path]:
    """Locates a shard's ``predictions.csv``.

    Args:
        run_dir: Shard output directory.

    Returns:
        The path, or ``None`` when the shard has not finished writing.
    """
    top = run_dir / "predictions.csv"
    if top.exists():
        return top
    nested = sorted(run_dir.glob("*/predictions.csv"))
    return nested[0] if nested else None


def read_shard(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    """Reads one shard's predictions.

    Args:
        path: Path to the shard's ``predictions.csv``.

    Returns:
        The header field names and the rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def merge(
    shard_dirs: List[Path],
    output: Path,
    expected: Optional[Path],
) -> Dict[str, object]:
    """Concatenates shard predictions after validating the partition.

    Args:
        shard_dirs: Shard output directories.
        output: Directory to write the merged ``predictions.csv`` into.
        expected: Optional evaluation-set JSON whose ``claim_id`` values
            the merged predictions must cover exactly.

    Returns:
        A summary of the merge.

    Raises:
        ValueError: If shards disagree on their columns, overlap on a
            claim identifier, or do not match ``expected``.
        FileNotFoundError: If any shard has no predictions file.
    """
    header: List[str] = []
    rows: List[Dict[str, str]] = []
    seen: Dict[str, str] = {}
    per_shard: Dict[str, int] = {}

    for run_dir in shard_dirs:
        path = resolve_predictions_csv(run_dir)
        if path is None:
            raise FileNotFoundError(f"No predictions.csv under {run_dir}")
        fieldnames, shard_rows = read_shard(path)
        if not header:
            header = fieldnames
        elif fieldnames != header:
            raise ValueError(
                f"Column mismatch in {path}: {fieldnames} != {header}"
            )
        for row in shard_rows:
            claim_id = row["claim_id"]
            if claim_id in seen:
                raise ValueError(
                    f"claim_id {claim_id} appears in both "
                    f"{seen[claim_id]} and {path}"
                )
            seen[claim_id] = str(path)
        per_shard[run_dir.name] = len(shard_rows)
        rows.extend(shard_rows)
        logger.info("%s: %d rows", path, len(shard_rows))

    if expected is not None:
        with expected.open(encoding="utf-8") as handle:
            want = {r["claim_id"] for r in json.load(handle)}
        missing = want - set(seen)
        extra = set(seen) - want
        if missing or extra:
            raise ValueError(
                f"Merged predictions do not match {expected}: "
                f"{len(missing)} missing, {len(extra)} unexpected"
            )
        logger.info(
            "Coverage verified against %s: %d rows", expected, len(want)
        )

    output.mkdir(parents=True, exist_ok=True)
    target = output / "predictions.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output": str(target),
        "total_rows": len(rows),
        "per_shard": per_shard,
        "coverage_verified": expected is not None,
    }
    with (output / "merge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description="Merges sharded inference predictions into one run dir."
    )
    parser.add_argument(
        "--shard-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Shard output directories to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write the merged predictions.csv into.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help=(
            "Evaluation-set JSON that the merged predictions must cover "
            "exactly; skipped when omitted."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Runs the shard merge."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        summary = merge(args.shard_dirs, args.output, args.expected)
    except (ValueError, FileNotFoundError) as error:
        logger.error("%s", error)
        sys.exit(1)
    logger.info("Merged %d rows into %s", summary["total_rows"], args.output)


if __name__ == "__main__":
    main()
