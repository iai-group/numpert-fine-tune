"""Stratified sampling of a gold-annotation seed set.

Samples records proportionally across (taxonomy_label, label) strata so
the gold seed mirrors the corpus distribution. Deterministic given a
seed.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List

_STRATA_KEYS = ("taxonomy_label", "label")


def _stratum_key(record: Dict[str, Any]) -> tuple:
    """Return the stratum key for a record."""
    return tuple(record.get(k) for k in _STRATA_KEYS)


def stratified_sample(
    data: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """Return a reproducible stratified sample of size up to `n`."""
    if n >= len(data):
        return list(data)

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in data:
        groups[_stratum_key(record)].append(record)

    total = len(data)
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []

    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        members = groups[key]
        take = round(n * len(members) / total)
        take = min(take, len(members))
        sample.extend(rng.sample(members, take))

    # Correct any rounding drift to hit exactly n.
    if len(sample) > n:
        sample = rng.sample(sample, n)
    elif len(sample) < n:
        chosen = {id(r) for r in sample}
        remaining = [r for r in data if id(r) not in chosen]
        sample.extend(rng.sample(remaining, n - len(sample)))

    return sample


def smoke_sample(
    data: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """Pick `n` records varying by document length and class.

    Splits the data into short / medium / long thirds by `doc` length and
    draws a class-stratified slice from each third, so a smoke run
    exercises both length and taxonomy/label variety.
    """
    if n >= len(data):
        return list(data)

    ranked = sorted(data, key=lambda r: len(r.get("doc", "")))
    third = len(ranked) // 3
    buckets = [
        ranked[:third],
        ranked[third : 2 * third],
        ranked[2 * third :],
    ]

    per = n // 3
    sample: List[Dict[str, Any]] = []
    for i, bucket in enumerate(buckets):
        take = per if i < 2 else n - 2 * per
        take = min(take, len(bucket))
        sample.extend(stratified_sample(bucket, take, seed + i))
    return sample
