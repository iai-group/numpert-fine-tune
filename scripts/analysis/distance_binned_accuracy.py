"""Accuracy binned by relative numerical distance of the perturbation.

For every perturbed test instance, extracts the numeric values from the
normalized (original) claim and its perturbed variant, computes the
relative distance of the changed quantity, and reports accuracy per
distance bin for the zero-shot baseline and the all-perturbations model
(q4_k_m in-distribution runs, as in the paper's main tables).

Distance definition per instance:
* No numeric change (hedge words, digit-to-word rewrites): r = 0.
* Value replacement v -> a: r = |a - v| / |v|.
* Range rewrite v -> [lo, hi]: r = 0 if lo <= v <= hi, otherwise the
  distance to the nearest bound divided by |v|.
* Multiple changed quantities: the maximum r over changed quantities.

Instances whose numbers cannot be aligned are reported as coverage loss.

Usage:
    python scripts/analysis/distance_binned_accuracy.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "data" / "perturbed_data" / "test_split"
RESULTS = REPO_ROOT / "results" / "inference" / "qwen3-8b-q4_k_m"
RUNS = {
    "baseline": RESULTS / "baseline" / "predictions.csv",
    "all_perturb": RESULTS
    / "Qwen3-8B-young-nightingale-manifests_all_perturb_no_f_type2_q4_k_m"
    / "predictions.csv",
}

BINS = (
    (-1.0, 0.0, "0 (value preserved)"),
    (0.0, 0.10, "(0, 10%]"),
    (0.10, 0.25, "(10%, 25%]"),
    (0.25, 0.50, "(25%, 50%]"),
    (0.50, 1.00, "(50%, 100%]"),
    (1.00, float("inf"), "> 100%"),
)

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(
    r"\b(" + "|".join(list(_UNITS) + list(_SCALES)) + r")(?:[\s-]+("
    + "|".join(list(_UNITS) + list(_SCALES))
    + r"))*\b",
    re.IGNORECASE,
)


def _parse_word_number(text: str) -> Optional[float]:
    """Parse a simple English number phrase ('fifty', 'twenty-five')."""
    tokens = re.split(r"[\s-]+", text.lower())
    total = 0.0
    current = 0.0
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _SCALES:
            current = (current or 1.0) * _SCALES[tok]
            total += current
            current = 0.0
        else:
            return None
    return total + current


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values (digits and simple word forms) from text."""
    values: List[Tuple[int, float]] = []
    for m in _NUM_RE.finditer(text):
        digits = m.group(0).replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        tail = text[m.end():].lstrip()
        for word, mult in _SCALES.items():
            if tail.lower().startswith(word):
                value *= mult
                break
        values.append((m.start(), value))
    stripped = _NUM_RE.sub(" ", text)
    for m in _WORD_RE.finditer(stripped):
        parsed = _parse_word_number(m.group(0))
        if parsed is not None:
            values.append((m.start(), parsed))
    return [v for _, v in sorted(values)]


def _interval_distance(v: float, lo: float, hi: float) -> float:
    """Relative distance from a value to an interval (0 if inside)."""
    if lo <= v <= hi:
        return 0.0
    gap = min(abs(v - lo), abs(v - hi))
    return gap / max(abs(v), 1e-9)


def relative_distance(
    original: str, perturbed: str, method: str
) -> Optional[float]:
    """Relative numerical distance between claim variants; None if
    the changed quantities cannot be aligned."""
    orig = extract_numbers(original)
    pert = extract_numbers(perturbed)
    removed = list(orig)
    added = list(pert)
    for v in list(removed):
        for a in list(added):
            if abs(a - v) <= 1e-9 * max(1.0, abs(v)):
                removed.remove(v)
                added.remove(a)
                break
    if not removed and not added:
        return 0.0
    if not removed:
        return None
    if method == "range" and len(added) == 2 * len(removed):
        added_sorted = sorted(added)
        pairs = [
            (added_sorted[2 * i], added_sorted[2 * i + 1])
            for i in range(len(removed))
        ]
        dists = []
        for v in sorted(removed):
            best = min(pairs, key=lambda p: _interval_distance(v, *p))
            dists.append(_interval_distance(v, *best))
            pairs.remove(best)
        return max(dists)
    if len(added) == len(removed):
        dists = []
        for v, a in zip(sorted(removed), sorted(added)):
            dists.append(abs(a - v) / max(abs(v), 1e-9))
        return max(dists)
    return None


def load_variants() -> Dict[Tuple[str, str, str], Optional[float]]:
    """Map (claim_id, source_file, perturbation_type) -> distance."""
    distances: Dict[Tuple[str, str, str], Optional[float]] = {}
    for path in sorted(TEST_DIR.glob("*.json")):
        label = path.stem.rsplit("_", 1)[1]  # True / False
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            original = item.get("normalized_claim") or item["claim"]
            method = item.get("perturbation_method", "")
            for key, value in item.items():
                if not key.startswith("perturb_type"):
                    continue
                ptype = key[: -len("_" + label)]
                if ptype == "perturb_type3_masked":
                    continue
                distances[(item["claim_id"], path.name, ptype)] = (
                    relative_distance(original, value, method)
                )
    return distances


def load_predictions(path: Path) -> Dict[Tuple[str, str, str], Tuple]:
    """Map instance key -> (gold, pred_or_None)."""
    out: Dict[Tuple[str, str, str], Tuple] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (
                row["claim_id"],
                row["source_file"],
                row["perturbation_type"],
            )
            gold = json.loads(row["label"])["label"]
            pred = (
                None
                if row["prediction"] == "PARSE_FAILURE"
                else json.loads(row["prediction"])["label"]
            )
            out[key] = (gold, pred)
    return out


def main() -> None:
    """Compute and print the distance-binned accuracy table."""
    distances = load_variants()
    runs = {name: load_predictions(p) for name, p in RUNS.items()}
    base = runs["baseline"]

    keys = [
        k
        for k in base
        if k[2] in ("perturb_type1", "perturb_type2", "perturb_type3_random")
    ]
    matched = [k for k in keys if k in distances]
    aligned = [k for k in matched if distances[k] is not None]
    print(
        f"perturbed instances: {len(keys)}; with variant text: "
        f"{len(matched)}; distance aligned: {len(aligned)} "
        f"({100 * len(aligned) / max(len(keys), 1):.1f}% coverage)"
    )

    stats = defaultdict(lambda: defaultdict(float))
    for key in aligned:
        r = distances[key]
        for lo, hi, label in BINS:
            if lo < r <= hi or (r == 0.0 and hi == 0.0):
                bin_label = label
                break
        s = stats[bin_label]
        s["n"] += 1
        for name in RUNS:
            gold, pred = runs[name][key]
            if pred is not None:
                s[f"{name}_valid"] += 1
                s[f"{name}_correct"] += pred == gold
            s[f"{name}_eff_correct"] += (
                pred is not None and pred == gold
            )

    header = (
        f"{'distance bin':22s} {'n':>5s} "
        f"{'base_acc':>9s} {'base_fail%':>10s} "
        f"{'allpert_acc':>11s} {'delta_pp':>9s}"
    )
    print(header)
    for lo, hi, label in BINS:
        s = stats.get(label)
        if not s:
            continue
        b_acc = 100 * s["baseline_correct"] / max(s["baseline_valid"], 1)
        b_fail = 100 * (1 - s["baseline_valid"] / s["n"])
        f_acc = 100 * s["all_perturb_correct"] / max(
            s["all_perturb_valid"], 1
        )
        print(
            f"{label:22s} {int(s['n']):5d} {b_acc:9.2f} {b_fail:10.1f} "
            f"{f_acc:11.2f} {f_acc - b_acc:+9.2f}"
        )
    out_json = {
        label: dict(stats[label]) for _, _, label in BINS if label in stats
    }
    out_path = REPO_ROOT / "results" / "rebuttal_stats" / "distance_bins.json"
    out_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
