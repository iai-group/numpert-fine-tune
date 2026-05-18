r"""Dataset split overview — per-perturbation-type size breakdown.

Prints a plain-text table and a LaTeX table showing the number of samples
per perturbation category (T→T, F→F, T→F) for each dataset split.

Usage::

    # All splits (default)
    python -m tests.sanity_checks.check_dataset_split_overview

    # Specific split(s)
    python -m tests.sanity_checks.check_dataset_split_overview \
        --splits train val

    # Custom data root
    python -m tests.sanity_checks.check_dataset_split_overview \
        --data_root data/perturbed_data
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.llm_finetune.utils.data_loading.dataset_balancer import (
    Sample,
    flatten_samples,
    load_files,
)

DATA_ROOT = Path("data/perturbed_data")
SEP = "=" * 80
THIN_SEP = "-" * 80

# Display order for methods
METHOD_ORDER = [
    "numeration",
    "approximation",
    "range",
    "masked",
    "random",
    "negative",
]

# LaTeX macro names per method
METHOD_LATEX: dict[str, str] = {
    "numeration": r"\numeration",
    "approximation": r"\approximation",
    "range": r"\range",
    "masked": r"\mask",
    "random": r"\random",
    "negative": r"\negnum",
}

# Human-readable display names
METHOD_DISPLAY: dict[str, str] = {
    "numeration": "Numeration",
    "approximation": "Approximation",
    "range": "Range",
    "masked": "Masking",
    "random": "Random",
    "negative": "Negation",
}

TRANSITION_TYPES = ["T→T", "T→F", "F→F"]
XMARK_PLAIN = "—"
XMARK_LATEX = r"\xmark"

# Methods that never produce T→T perturbations
NO_T_TO_T_METHODS = frozenset({"masked", "random", "negative"})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(split: str, data_root: Path) -> list[Sample]:
    """Load and flatten all JSON files for the given split."""
    split_dir = data_root / f"{split}_split"
    file_paths: list[str | Path] = sorted(split_dir.glob("*.json"))
    if not file_paths:
        raise FileNotFoundError(
            f"No JSON files found in {split_dir}. "
            "Check --data_root and --splits."
        )
    file_data = load_files(file_paths)
    return flatten_samples(file_data)


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------


def count_by_method_and_type(
    samples: list[Sample],
) -> dict[str, dict[str, int]]:
    """Return {method: {transition_type: count}} for perturbed samples."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        if s.is_perturbed and s.perturbation_method is not None:
            counts[s.perturbation_method][s.type] += 1
    return {method: dict(type_counts) for method, type_counts in counts.items()}


def count_originals(samples: list[Sample]) -> dict[str, int]:
    """Count unique original claims by label (deduplicated by claim_id)."""
    seen: set[str] = set()
    label_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        if not s.is_perturbed and s.source_claim_id not in seen:
            seen.add(s.source_claim_id)
            label_counts[s.label] += 1
    return dict(label_counts)


# ---------------------------------------------------------------------------
# Plain-text output
# ---------------------------------------------------------------------------


def print_text_table(
    split: str,
    counts: dict[str, dict[str, int]],
    orig_counts: dict[str, int],
) -> None:
    """Print a plain-text overview table for one split."""
    n_true = orig_counts.get("True", 0)
    n_false = orig_counts.get("False", 0)

    print(f"\n{SEP}")
    print(f"  SPLIT: {split.upper()}")
    print(f"  Unique original claims — True: {n_true}  False: {n_false}")
    print(SEP)

    col_w = 14
    header = (
        f"  {'Category':<18s}"
        f"  {'T→T':>{col_w}}"
        f"  {'T→F':>{col_w}}"
        f"  {'F→F':>{col_w}}"
        f"  {'Total':>{col_w}}"
    )
    print(header)
    print(f"  {THIN_SEP}")

    all_methods = [
        m for m in METHOD_ORDER if m in counts or m in METHOD_DISPLAY
    ]
    for method in all_methods:
        method_counts = counts.get(method, {})
        t_to_t = method_counts.get("T→T", 0)
        t_to_f = method_counts.get("T→F", 0)
        f_to_f = method_counts.get("F→F", 0)
        total = t_to_t + t_to_f + f_to_f

        t_to_t_str = XMARK_PLAIN if method in NO_T_TO_T_METHODS else str(t_to_t)
        display = METHOD_DISPLAY.get(method, method)

        print(
            f"  {display:<18s}"
            f"  {t_to_t_str:>{col_w}}"
            f"  {t_to_f:>{col_w}}"
            f"  {f_to_f:>{col_w}}"
            f"  {total:>{col_w}}"
        )

    print(f"  {THIN_SEP}")

    # Totals row
    total_t_to_t = sum(
        c.get("T→T", 0) for m, c in counts.items() if m not in NO_T_TO_T_METHODS
    )
    total_t_to_f = sum(c.get("T→F", 0) for c in counts.values())
    total_f_to_f = sum(c.get("F→F", 0) for c in counts.values())
    grand_total = total_t_to_t + total_t_to_f + total_f_to_f

    print(
        f"  {'TOTAL':<18s}"
        f"  {total_t_to_t:>{col_w}}"
        f"  {total_t_to_f:>{col_w}}"
        f"  {total_f_to_f:>{col_w}}"
        f"  {grand_total:>{col_w}}"
    )


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def render_latex_table(
    split: str,
    counts: dict[str, dict[str, int]],
    orig_counts: dict[str, int],
    label: Optional[str] = None,
) -> str:
    """Render a LaTeX table for one split and return it as a string."""
    n_true = orig_counts.get("True", 0)
    n_false = orig_counts.get("False", 0)
    table_label = label or f"tab:dataset_overview_{split}"

    lines: list[str] = []
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage{pifont}")
    lines.append(r"% \newcommand{\cmark}{\ding{51}}")
    lines.append(r"% \newcommand{\xmark}{\ding{55}}")
    lines.append("")
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\small The number of samples per perturbation type that"
        r" remain \texttt{True} (T$\rightarrow$T), switch from \texttt{True}"
        r" to \texttt{False} (T$\rightarrow$F), or remain \texttt{False}"
        rf" (F$\rightarrow$F) in the \textbf{{{split}}} split."
        rf" Total unique original claims: {n_true} \texttt{{True}}"
        rf" and {n_false} \texttt{{False}}.}}"
    )
    lines.append(rf"\label{{{table_label}}}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Category}"
        r" & \textbf{T$\rightarrow$T}"
        r" & \textbf{T$\rightarrow$F}"
        r" & \textbf{F$\rightarrow$F}"
        r" & \textbf{Total} \\"
    )
    lines.append(r"\midrule")

    all_methods = [
        m for m in METHOD_ORDER if m in counts or m in METHOD_DISPLAY
    ]
    for method in all_methods:
        method_counts = counts.get(method, {})
        t_to_t = method_counts.get("T→T", 0)
        t_to_f = method_counts.get("T→F", 0)
        f_to_f = method_counts.get("F→F", 0)
        total = t_to_t + t_to_f + f_to_f

        t_to_t_cell = (
            XMARK_LATEX if method in NO_T_TO_T_METHODS else str(t_to_t)
        )
        latex_cmd = METHOD_LATEX.get(method, method)

        lines.append(
            rf"\textbf{{{latex_cmd}}}"
            f" & {t_to_t_cell}"
            f" & {t_to_f}"
            f" & {f_to_f}"
            f" & {total} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Print per-perturbation-type sample counts for each dataset split."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which splits to process (default: all three).",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root directory for perturbed data (default: {DATA_ROOT}).",
    )
    parser.add_argument(
        "--latex_only",
        action="store_true",
        help="Print only the LaTeX tables (skip plain-text tables).",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Print only the plain-text tables (skip LaTeX output).",
    )
    args = parser.parse_args()

    for split in args.splits:
        samples = load_split(split, args.data_root)
        counts = count_by_method_and_type(samples)
        orig_counts = count_originals(samples)

        if not args.latex_only:
            print_text_table(split, counts, orig_counts)

        if not args.text_only:
            latex = render_latex_table(split, counts, orig_counts)
            print(f"\n% ---- LaTeX table: {split} split ----")
            print(latex)

    print(f"\n{SEP}")
    print("  DONE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
