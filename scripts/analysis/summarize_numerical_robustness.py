"""Assembles the cross-dataset numerical-robustness comparison table.

Three training regimes are compared on two test sets that probe different
numerical-robustness competencies:

    QuanTemp  synthetic CLAIM-side perturbations (the paper's setting)
    VitaminC  naturally occurring EVIDENCE-side revisions from Wikipedia

For VitaminC the table also reports *pair accuracy*: the share of
contrastive pairs where a model labels both snapshots of one revision
correctly. That is the natural analogue of the paper's label-flipping
evaluation, and it separates a model that compares quantities from one
that merely checks whether they are close.

Every number is read from the runs' own ``metrics.json`` and
``predictions.csv``, so the table cannot drift from the artifacts.

Usage:
    python scripts/analysis/summarize_numerical_robustness.py \
        --output results/numerical_robustness_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vitaminc_comparison import (  # noqa: E402
    build_pairs,
    load_pair_index,
    load_run,
    macro_f1,
)

logger = logging.getLogger(__name__)

VITAMINC_DATASET = Path("data/vitaminc/vitaminc_test_numerical.json")


@dataclass
class Entry:
    """One model's run directories on each test set.

    Attributes:
        label: Human-readable model description for the table.
        vitaminc: Run directory for the VitaminC test evaluation.
        quantemp: Run directory for the QuanTemp test evaluation.
        note: Optional qualifier shown beneath the table.
    """

    label: str
    vitaminc: Optional[Path] = None
    quantemp: Optional[Path] = None
    note: str = ""


VITAMINC_BASE = Path("results/inference/vitaminc/qwen3-8b-q4_k_m")
QUANTEMP_BASE = Path("results/inference/qwen3-8b-q4_k_m")

ENTRIES: List[Entry] = [
    Entry(
        label="Base Qwen3-8B, zero-shot",
        vitaminc=VITAMINC_BASE / "baseline_merged",
        note="reasons with chain-of-thought; 5.9% parse failures",
    ),
    Entry(
        label="QuanTemp-tuned",
        vitaminc=VITAMINC_BASE / "young_nightingale",
        quantemp=(
            QUANTEMP_BASE
            / "Qwen3-8B-young-nightingale-manifests_all_perturb_no_f_type2"
            "_q4_k_m"
        ),
    ),
    Entry(
        label="QuanTemp-tuned + threshold prompt",
        vitaminc=VITAMINC_BASE / "young_nightingale_threshold",
        note="prompt selected on the VitaminC dev split",
    ),
    Entry(
        label="QuanTemp-tuned + structured CoT",
        vitaminc=VITAMINC_BASE / "young_nightingale_structured_cot",
    ),
    Entry(
        label="QuanTemp-tuned + PAP",
        vitaminc=VITAMINC_BASE / "young_nightingale_pap",
    ),
    Entry(
        label="VitaminC-trained",
        vitaminc=VITAMINC_BASE / "vitaminc_only",
        quantemp=(
            QUANTEMP_BASE / "Qwen3-8B-daring-ant-evolves_vitaminc_only_q4_k_m"
        ),
    ),
    Entry(
        label="Mixed QuanTemp + VitaminC",
        vitaminc=VITAMINC_BASE / "mixed_quantemp_vitaminc",
        quantemp=(QUANTEMP_BASE / "Qwen3-8B-mixed_quantemp_vitaminc_q4_k_m"),
    ),
]


@dataclass
class Cell:
    """Metrics for one model on one test set.

    Attributes:
        accuracy: Accuracy over all instances, as a percentage.
        macro_f1: Macro-F1 as a percentage, or ``None`` if unavailable.
        total: Number of instances evaluated.
        parse_failures: Count of unparseable responses.
        pair_accuracy: Contrastive-pair accuracy, VitaminC only.
        extra: Any per-group accuracies worth reporting.
    """

    accuracy: float
    macro_f1: Optional[float]
    total: int
    parse_failures: int
    pair_accuracy: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


def find_metrics(run_dir: Path) -> Optional[Path]:
    """Locates a run's ``metrics.json``, including one nested level.

    Args:
        run_dir: Directory to search.

    Returns:
        The path, or ``None`` when the run has not produced metrics.
    """
    direct = run_dir / "metrics.json"
    if direct.exists():
        return direct
    nested = sorted(run_dir.glob("*/metrics.json"))
    return nested[0] if nested else None


def read_cell(
    run_dir: Path,
    pair_index: Optional[Dict[str, tuple]] = None,
) -> Optional[Cell]:
    """Reads one run's metrics, optionally adding pair accuracy.

    Args:
        run_dir: The run's output directory.
        pair_index: When given, contrastive-pair accuracy is computed from
            the run's predictions using this (case, claim) index.

    Returns:
        The assembled cell, or ``None`` when the run is absent.
    """
    metrics_path = find_metrics(run_dir)

    run = None
    if pair_index is not None or metrics_path is None:
        try:
            run = load_run(run_dir)
        except (FileNotFoundError, ValueError) as error:
            logger.warning("No predictions for %s: %s", run_dir, error)

    if metrics_path is None:
        # A merged run (assembled from shards) carries predictions but no
        # metrics of its own; derive the overall figures from them.
        if run is None:
            return None
        gold = np.array([i.gold for i in run.values()], dtype=bool)
        correct_flags = np.array([i.correct for i in run.values()], dtype=bool)
        overall = {
            "accuracy": float(np.mean(correct_flags)),
            "f1_macro": macro_f1(gold, correct_flags),
            "total": len(run),
            "parse_failures": sum(1 for i in run.values() if not i.parsed),
        }
        per_type: Dict[str, dict] = {}
    else:
        document = json.load(open(metrics_path, encoding="utf-8"))
        overall = document["overall"]
        per_type = document.get("per_perturbation_type", {})

    pair_accuracy = None
    if pair_index is not None and run:
        pairs = build_pairs(sorted(run), pair_index, run)
        if pairs:
            correct = sum(
                1 for a, b in pairs if run[a].correct and run[b].correct
            )
            pair_accuracy = 100.0 * correct / len(pairs)

    extra = {
        name: 100.0 * group["accuracy"]
        for name, group in per_type.items()
        if "accuracy" in group
    }
    return Cell(
        accuracy=100.0 * overall["accuracy"],
        macro_f1=100.0 * overall["f1_macro"]
        if overall.get("f1_macro") is not None
        else None,
        total=overall["total"],
        parse_failures=overall.get("parse_failures", 0),
        pair_accuracy=pair_accuracy,
        extra=extra,
    )


def format_number(value: Optional[float]) -> str:
    """Formats a percentage, or an em dash when absent."""
    return "—" if value is None else f"{value:.2f}"


def render(rows: List[tuple]) -> str:
    """Renders the assembled rows as a Markdown report.

    Args:
        rows: Tuples of (entry, VitaminC cell, QuanTemp cell).

    Returns:
        The report text.
    """
    lines = [
        "# Numerical robustness across two test sets",
        "",
        "`VitaminC` probes naturally occurring evidence-side numerical "
        "revisions; `QuanTemp` probes synthetic claim-side perturbations. "
        "Pair accuracy requires both snapshots of one revision to be "
        "labelled correctly.",
        "",
        "| Model | VitaminC acc | VitaminC macro-F1 | VitaminC pair acc "
        "| VitaminC parse fails | QuanTemp acc | QuanTemp macro-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    notes = []
    for entry, vc, qt in rows:
        lines.append(
            f"| {entry.label} | "
            f"{format_number(vc.accuracy) if vc else '—'} | "
            f"{format_number(vc.macro_f1) if vc else '—'} | "
            f"{format_number(vc.pair_accuracy) if vc else '—'} | "
            f"{vc.parse_failures if vc else '—'} | "
            f"{format_number(qt.accuracy) if qt else '—'} | "
            f"{format_number(qt.macro_f1) if qt else '—'} |"
        )
        if entry.note:
            notes.append(f"- {entry.label}: {entry.note}")

    sizes = {("VitaminC", vc.total) for _, vc, _ in rows if vc} | {
        ("QuanTemp", qt.total) for _, _, qt in rows if qt
    }
    lines += [
        "",
        "Instances per test set: "
        + ", ".join(f"{name} {size:,}" for name, size in sorted(sizes)),
        "",
    ]
    if notes:
        lines += ["Notes:", *notes, ""]

    quantemp_rows = [(e, qt) for e, _, qt in rows if qt and qt.extra]
    if quantemp_rows:
        types = sorted({t for _, qt in quantemp_rows for t in qt.extra})
        lines += [
            "## QuanTemp accuracy by perturbation type",
            "",
            "| Model | " + " | ".join(types) + " |",
            "|---" * (len(types) + 1) + "|",
        ]
        for entry, qt in quantemp_rows:
            cells = " | ".join(format_number(qt.extra.get(t)) for t in types)
            lines.append(f"| {entry.label} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Assembles the cross-dataset numerical-robustness table from "
            "the evaluation runs' own metrics and predictions."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/numerical_robustness_summary.md"),
        help="Destination for the Markdown report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=VITAMINC_DATASET,
        help="VitaminC evaluation subset, used for pair grouping.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Assembles and writes the summary table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    pair_index = (
        load_pair_index(args.dataset) if args.dataset.exists() else None
    )
    if pair_index is None:
        logger.warning(
            "VitaminC dataset %s missing; pair accuracy omitted",
            args.dataset,
        )

    rows = []
    for entry in ENTRIES:
        vc = (
            read_cell(entry.vitaminc, pair_index)
            if entry.vitaminc is not None
            else None
        )
        qt = read_cell(entry.quantemp) if entry.quantemp is not None else None
        if vc is None and qt is None:
            logger.info("Skipping %s (no runs found)", entry.label)
            continue
        rows.append((entry, vc, qt))

    report = render(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
