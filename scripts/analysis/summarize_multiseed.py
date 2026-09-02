"""Summarize multi-seed replication runs as mean and std accuracy.

Reads ``results/inference/multiseed/<experiment>/seed_*/metrics.json``
(written by scripts/train/slurm_multiseed_all_perturb.sh) and prints the
overall and per-perturbation-type accuracy across seeds.

Usage:
    python scripts/analysis/summarize_multiseed.py \
        [results/inference/multiseed/qwen3_1_7b_all_perturb_types_no_f_type2]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

DEFAULT_DIR = Path(
    "results/inference/multiseed/qwen3_1_7b_all_perturb_types_no_f_type2"
)


def main() -> None:
    """Print mean/std accuracies over the available seed runs."""
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    metric_files = sorted(base.glob("seed_*/metrics.json"))
    if not metric_files:
        sys.exit(f"No seed_*/metrics.json found under {base}")

    rows: dict[str, list[float]] = {}
    for path in metric_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = path.parent.name
        rows.setdefault("_seeds", []).append(float(seed.split("_")[1]))
        rows.setdefault("overall", []).append(
            100.0 * data["overall"]["accuracy"]
        )
        for key, entry in data.get("per_perturbation_type", {}).items():
            rows.setdefault(key, []).append(100.0 * entry["accuracy"])

    seeds = [int(s) for s in rows.pop("_seeds")]
    print(f"Runs: {len(metric_files)} (seeds: {seeds})")
    print(f"{'scope':28s} {'mean':>8s} {'std':>7s}  values")
    for key, values in rows.items():
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        vals = ", ".join(f"{v:.2f}" for v in values)
        print(
            f"{key:28s} {statistics.mean(values):8.2f} {std:7.2f}  [{vals}]"
        )


if __name__ == "__main__":
    main()
