"""Samples VitaminC training and validation sets for continued tuning.

The VitaminC training subset produced by ``prepare_vitaminc.py`` holds
about 78k instances, far more than is needed to adapt an already
fine-tuned checkpoint and more than the QuanTemp training data it is
being compared against.  This module draws a smaller sample sized to be
comparable to QuanTemp, under three constraints:

    1. Sampling is by *case*, never by instance, so the two snapshots of
       a revision always land in the same split.  Splitting a pair would
       destroy the contrastive signal that makes VitaminC useful.
    2. Cases that carry both a supporting and a refuting instance are
       preferred, since those are the label-flipping examples that
       correspond to the paper's T->F training pairs.
    3. The validation set is carved out of the *training* split, using
       cases disjoint from the sampled training cases.  The official dev
       split is deliberately left untouched so that it remains usable for
       prompt selection without contaminating model selection.

Usage:
    python scripts/data/sample_vitaminc_training.py \
        --input data/vitaminc/vitaminc_train_numerical.json \
        --train-size 16000 \
        --val-size 2000
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SEED = 3407


def group_by_case(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups instances by their VitaminC case identifier.

    Args:
        records: Instances produced by ``prepare_vitaminc.py``.

    Returns:
        Mapping from case identifier to its instances.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[record["vitaminc_case_id"]].append(record)
    return dict(grouped)


def order_cases(
    grouped: Dict[str, List[Dict[str, Any]]],
    seed: int,
) -> List[str]:
    """Orders case identifiers, contrastive cases first.

    Args:
        grouped: Cases as returned by :func:`group_by_case`.
        seed: Seed for the shuffling within each group.

    Returns:
        Case identifiers, those carrying both labels first, each block
        shuffled deterministically.
    """
    contrastive: List[str] = []
    single: List[str] = []
    for case_id, instances in grouped.items():
        labels = {r["label"] for r in instances}
        if {"True", "False"} <= labels:
            contrastive.append(case_id)
        else:
            single.append(case_id)
    rng = random.Random(seed)
    rng.shuffle(contrastive)
    rng.shuffle(single)
    logger.info(
        "Cases: %d contrastive, %d single-label",
        len(contrastive),
        len(single),
    )
    return contrastive + single


def take_cases(
    ordered: List[str],
    grouped: Dict[str, List[Dict[str, Any]]],
    target: int,
    start: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Collects whole cases until the instance target is reached.

    Args:
        ordered: Case identifiers in the order they should be consumed.
        grouped: Cases as returned by :func:`group_by_case`.
        target: Desired number of instances.
        start: Index in ``ordered`` at which to begin.

    Returns:
        The collected instances and the index just past the last case
        consumed, so a caller can continue without overlap.
    """
    selected: List[Dict[str, Any]] = []
    index = start
    while index < len(ordered) and len(selected) < target:
        selected.extend(grouped[ordered[index]])
        index += 1
    return selected, index


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes distribution statistics for one sampled split.

    Args:
        records: The sampled instances.

    Returns:
        Instance and case counts plus the label distribution.
    """
    cases = collections.Counter(r["vitaminc_case_id"] for r in records)
    contrastive = 0
    by_case: Dict[str, set] = collections.defaultdict(set)
    for record in records:
        by_case[record["vitaminc_case_id"]].add(record["label"])
    for labels in by_case.values():
        if {"True", "False"} <= labels:
            contrastive += 1
    return {
        "instances": len(records),
        "cases": len(cases),
        "contrastive_cases": contrastive,
        "labels": dict(collections.Counter(r["label"] for r in records)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Samples case-disjoint VitaminC training and validation sets."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/vitaminc/vitaminc_train_numerical.json"),
        help="Full VitaminC training subset to sample from.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=16000,
        help="Approximate number of training instances (default: 16000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Approximate number of validation instances (default: 2000).",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/vitaminc/vitaminc_train_sample.json"),
        help="Destination for the sampled training set.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/vitaminc/vitaminc_val_sample.json"),
        help="Destination for the sampled validation set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Runs the sampling end to end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    with args.input.open(encoding="utf-8") as handle:
        records = json.load(handle)
    logger.info("Loaded %d instances from %s", len(records), args.input)

    grouped = group_by_case(records)
    ordered = order_cases(grouped, args.seed)

    train, cursor = take_cases(ordered, grouped, args.train_size)
    val, _ = take_cases(ordered, grouped, args.val_size, start=cursor)

    train_cases = {r["vitaminc_case_id"] for r in train}
    val_cases = {r["vitaminc_case_id"] for r in val}
    overlap = train_cases & val_cases
    if overlap:
        raise ValueError(
            f"Train and validation share {len(overlap)} cases; "
            "sampling is not disjoint."
        )

    for split_name, split_records, destination in (
        ("train", train, args.train_output),
        ("validation", val, args.val_output),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(split_records, handle, ensure_ascii=False, indent=2)
        logger.info(
            "%s -> %s: %s", split_name, destination, summarize(split_records)
        )


if __name__ == "__main__":
    main()
