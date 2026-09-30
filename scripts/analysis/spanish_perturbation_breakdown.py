"""Breaks the CLEF Spanish results down by perturbation type.

The cross-dataset table reports CLEF Spanish only at the level of the
transition groups (Original, T1, T2, T3), which aggregate over the
label transition. On a split that is 95.8% ``False`` this hides the
effect that matters: the T2 column pools 553 hard T->F instances with
5,578 easy F->F ones, so a 24-point gain on label-flipping appears as a
two-point gain overall.

This module recomputes the split per perturbation type *within* each
label transition, writes a LaTeX table in the style of the QuanTemp
per-direction tables, and prints the same figures for inspection.

Accuracy is computed on each model's parseable responses, matching the
protocol of the cross-dataset table; parse-failure rates are reported
separately because the baseline leaves a large share unparseable.

Usage:
    python scripts/analysis/spanish_perturbation_breakdown.py \
        --output tables/results_q4_k_m/spanish_perturbation.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RUN_ROOT = Path("results/inference/spanish-test/qwen3-8b-q4_k_m")

# Perturbation type marking an unperturbed claim.
ORIGINAL_TYPE = "original_normalized"

MODELS: List[Tuple[str, str]] = [
    ("baseline", r"\mbase"),
    ("no_approx", r"\mnoapproximation"),
    ("young_nightingale", r"\mallperturb"),
]

# Source-file stem -> column label, in table order.
METHODS: List[Tuple[str, str]] = [
    ("approximation", r"\textbf{Approx.}"),
    ("numeration", r"\textbf{Num.}"),
    ("range", r"\textbf{Range}"),
    ("negative", r"\textbf{Neg.}"),
    ("obfuscation", r"\textbf{Rand.}"),
]

DIRECTIONS: List[Tuple[str, str]] = [
    ("T->F", r"\ttf\ (label-flipping)"),
    ("T->T", r"\ttt\ (label-preserving)"),
    ("F->F", r"\ftf\ (already false)"),
]


def parse_label(cell: str) -> Optional[bool]:
    """Parses a ``'{"label": true}'`` cell into a boolean.

    Args:
        cell: Raw CSV cell contents.

    Returns:
        The verdict, or ``None`` when the cell is a parse failure.
    """
    try:
        value = json.loads(cell)["label"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    return value if isinstance(value, bool) else None


def find_predictions(subdir: str) -> Path:
    """Locates a run's predictions file beneath its output directory.

    The run directories are dot-prefixed, so ordinary globbing skips
    them; this shells out to ``find`` instead.

    Args:
        subdir: Model subdirectory under the Spanish run root.

    Returns:
        Path to that run's ``predictions.csv``.

    Raises:
        FileNotFoundError: If no predictions file exists.
    """
    found = subprocess.run(
        ["find", str(RUN_ROOT / subdir), "-name", "predictions.csv"],
        capture_output=True,
        text=True,
    ).stdout.split()
    if not found:
        raise FileNotFoundError(f"No predictions.csv under {RUN_ROOT/subdir}")
    return Path(found[0])


def method_of(row: Dict[str, str]) -> str:
    """Returns the perturbation method a row came from."""
    return row["source_file"].rsplit("_", 1)[0]


def direction_of(row: Dict[str, str]) -> str:
    """Returns the gold label transition for a row.

    The original label is carried by the source filename and the
    perturbed label by the gold column, so the transition is read off
    the pair rather than assumed from the perturbation type.

    Args:
        row: One prediction record.

    Returns:
        One of ``"T->T"``, ``"T->F"`` or ``"F->F"``.
    """
    original = "T" if row["source_file"].endswith("_True.json") else "F"
    gold = "T" if parse_label(row["label"]) else "F"
    return f"{original}->{gold}"


def accuracy(rows: List[Dict[str, str]]) -> Tuple[Optional[float], int, int]:
    """Computes accuracy on parseable responses.

    Args:
        rows: Prediction records to score.

    Returns:
        Accuracy as a percentage (``None`` when nothing is parseable),
        the number of instances, and the number of parse failures.
    """
    valid = [r for r in rows if parse_label(r["prediction"]) is not None]
    if not valid:
        return None, len(rows), len(rows)
    correct = sum(
        1
        for r in valid
        if parse_label(r["prediction"]) == parse_label(r["label"])
    )
    return 100.0 * correct / len(valid), len(rows), len(rows) - len(valid)


def load_runs() -> Dict[str, List[Dict[str, str]]]:
    """Loads every model's Spanish predictions, excluding originals.

    Unperturbed rows carry the perturbation type
    ``original_normalized`` but keep the source file of the method they
    were drawn from, so leaving them in would count each original claim
    under whichever method sampled it and inflate the Numeration and
    Range columns. This table reports perturbed instances only; the
    originals are reported separately by the cross-dataset table.

    Returns:
        Each model's perturbed prediction rows, keyed by subdirectory.
    """
    runs = {}
    for subdir, _ in MODELS:
        path = find_predictions(subdir)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["perturbation_type"] != ORIGINAL_TYPE
            ]
        runs[subdir] = rows
        logger.info("%s: %d perturbed rows from %s", subdir, len(rows), path)
    return runs


def fmt(value: Optional[float], delta: Optional[float] = None) -> str:
    """Formats a cell, optionally with a coloured delta superscript."""
    if value is None:
        return "--"
    if delta is None:
        return f"{value:.2f}"
    colour = "numpertdeltaimprove" if delta >= 0 else "numpertdeltadecline"
    sign = "+" if delta >= 0 else "-"
    return (
        f"{value:.2f}\\textsuperscript{{\\textcolor{{{colour}}}"
        f"{{${sign}${abs(delta):.2f}}}}}"
    )


def render(runs: Dict[str, List[Dict[str, str]]]) -> str:
    """Renders the per-perturbation breakdown as a LaTeX table."""
    lines = [
        "% Generated by scripts/analysis/spanish_perturbation_breakdown.py",
        "% Requires booktabs, xcolor.",
        "\\begin{table*}[ht!]",
        "\\centering",
        "\\definecolor{numpertdeltaimprove}{RGB}{0,100,0}",
        "\\definecolor{numpertdeltadecline}{RGB}{180,35,24}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|ccccc|c}",
        "\\toprule",
        "\\textbf{Model} & "
        + " & ".join(label for _, label in METHODS)
        + " & \\textbf{Micro Avg.} \\\\",
    ]
    for key, pretty in DIRECTIONS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{pretty}}}}} \\\\")
        lines.append("\\midrule")
        base_cells: Dict[str, Optional[float]] = {}
        base_micro: Optional[float] = None
        for subdir, macro in MODELS:
            rows = [r for r in runs[subdir] if direction_of(r) == key]
            cells = []
            for method, _ in METHODS:
                sel = [r for r in rows if method_of(r) == method]
                acc, _, _ = accuracy(sel)
                if subdir == "baseline":
                    base_cells[method] = acc
                    cells.append(fmt(acc))
                else:
                    ref = base_cells.get(method)
                    d = None if (acc is None or ref is None) else acc - ref
                    cells.append(fmt(acc, d))
            micro, _, _ = accuracy(rows)
            if subdir == "baseline":
                base_micro = micro
                micro_cell = fmt(micro)
            else:
                d = (
                    None
                    if (micro is None or base_micro is None)
                    else micro - base_micro
                )
                micro_cell = fmt(micro, d)
            lines.append(
                f"{macro} & " + " & ".join(cells) + f" & {micro_cell} \\\\"
            )
        counts = []
        for method, _ in METHODS:
            n = len(
                [
                    r
                    for r in runs["young_nightingale"]
                    if direction_of(r) == key and method_of(r) == method
                ]
            )
            counts.append(str(n) if n else "--")
        total = len(
            [r for r in runs["young_nightingale"] if direction_of(r) == key]
        )
        lines.append(
            "\\textit{(n)} & "
            + " & ".join(f"\\textit{{{c}}}" for c in counts)
            + f" & \\textit{{{total}}} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\caption{\\footnotesize CLEF 2025 Spanish accuracy (\\%) by "
        "perturbation type \\emph{within} each label transition. "
        "Deltas are against \\mbase\\ in the same column. Accuracy is "
        "computed on each model's parseable responses, as in "
        "Table~\\ref{tab:generalization_results}. \\textbf{Neg.} and "
        "\\textbf{Rand.} produce \\texttt{False} labels by construction "
        "and so cannot appear under \\ttt. Reporting within transitions "
        "separates the 717 label-flipping instances from the 15{,}059 "
        "\\ftf\\ ones that dominate this split; the aggregated columns "
        "of Table~\\ref{tab:generalization_results} pool them and "
        "understate the label-flipping gain by an order of magnitude.}",
        "\\label{tab:spanish_perturbation}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def summarize(runs: Dict[str, List[Dict[str, str]]]) -> None:
    """Prints the breakdown and the parse-failure rates."""
    for key, pretty in DIRECTIONS:
        print(f"\n=== {key} ===")
        for subdir, macro in MODELS:
            rows = [r for r in runs[subdir] if direction_of(r) == key]
            cells = []
            for method, _ in METHODS:
                acc, _, _ = accuracy(
                    [r for r in rows if method_of(r) == method]
                )
                cells.append(
                    f"{method[:6]}={'--' if acc is None else f'{acc:6.2f}'}"
                )
            micro, n, pf = accuracy(rows)
            print(
                f"  {macro:20s} "
                + " ".join(cells)
                + f"  micro={micro:6.2f}  n={n}  parse_fail={pf}"
                f" ({100*pf/n:.1f}%)"
            )


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Breaks the CLEF Spanish results down by perturbation type "
            "within each label transition."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table to.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Computes the breakdown and optionally writes the LaTeX table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    runs = load_runs()
    summarize(runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(runs), encoding="utf-8")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
