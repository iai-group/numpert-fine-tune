#!/usr/bin/env python3
"""Compare split metrics groups across model directories.

This script scans a root directory recursively for ``metrics.json`` files,
extracts entries from ``per_file_and_type`` whose group key contains
``_True.json`` or ``_False.json``, and exports cross-model comparison CSV files.

It can also generate optional charts (PNG) if matplotlib is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def model_sort_key(model_name: str) -> Tuple[int, str]:
    """Sort models by requested priority, then alphabetically.

    Priority order:
    1) names containing "baseline"
    2) names containing "normalized"
    3) names containing "num"
    4) names containing "approx"
    5) names containing "range"
    6) all remaining names
    """
    name = model_name.lower()
    if "baseline" in name:
        return (0, name)
    if "normalized" in name:
        return (1, name)
    if "num" in name:
        return (2, name)
    if "approx" in name:
        return (3, name)
    if "range" in name:
        return (4, name)
    return (5, name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare accuracies and parse failures for *_True.json and "
            "*_False.json groups "
            "across model metrics.json files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing model subdirectories and metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV files (and optional charts) are written",
    )
    parser.add_argument(
        "--metrics-pattern",
        type=str,
        default="**/metrics.json",
        help=(
            "Glob pattern (relative to input-dir) used to find metrics files. "
            "Default: **/metrics.json"
        ),
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Create PNG charts in output-dir/charts",
    )
    parser.add_argument(
        "--include-normalized-in-charts",
        action="store_true",
        help=(
            "Include normalized baseline metrics "
            "(per_perturbation_type=original_normalized) in chart outputs"
        ),
    )
    parser.add_argument(
        "--model-name-from",
        choices=["parent", "stem"],
        default="parent",
        help=(
            "How to infer model name: 'parent' uses metrics.json parent "
            "folder name, 'stem' uses metrics filename stem"
        ),
    )
    return parser.parse_args()


def discover_metrics_files(root: Path, pattern: str) -> List[Path]:
    """Return sorted list of metrics files under root matching pattern."""
    return sorted(p for p in root.glob(pattern) if p.is_file())


def infer_model_name(metrics_path: Path, strategy: str) -> str:
    """Infer model name from metrics path using selected strategy."""
    if strategy == "stem":
        return metrics_path.stem
    return metrics_path.parent.name


def safe_float(value: Any) -> Optional[float]:
    """Convert value to float if possible, else return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Convert value to int if possible, else return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float_or_nan(value: Any) -> float:
    """Return float value or NaN when conversion is not possible."""
    maybe_value = safe_float(value)
    if maybe_value is None:
        return float("nan")
    return maybe_value


def extract_split_groups(
    metrics_data: Dict,
    split_name: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract per_file_and_type entries for a specific split (True/False)."""
    groups = metrics_data.get("per_file_and_type", {})
    out: Dict[str, Dict[str, Optional[float]]] = {}
    needle = f"_{split_name}.json"

    for group_name, payload in groups.items():
        if needle not in group_name:
            continue

        out[group_name] = {
            "accuracy": safe_float(payload.get("accuracy")),
            "parse_failures": safe_int(payload.get("parse_failures")),
            "total": safe_int(payload.get("total")),
            "valid": safe_int(payload.get("valid")),
            "correct": safe_int(payload.get("correct")),
        }
    return out


def extract_normalized_baseline(
    metrics_data: Dict,
) -> Optional[Dict[str, Optional[float]]]:
    """Extract original_normalized baseline from per_perturbation_type."""
    perturb_types = metrics_data.get("per_perturbation_type", {})
    baseline = perturb_types.get("original_normalized")

    if baseline is None:
        return None

    return {
        "accuracy": safe_float(baseline.get("accuracy")),
        "parse_failures": safe_int(baseline.get("parse_failures")),
        "total": safe_int(baseline.get("total")),
        "valid": safe_int(baseline.get("valid")),
        "correct": safe_int(baseline.get("correct")),
    }


def build_rows(
    metrics_files: Iterable[Path],
    input_dir: Path,
    model_name_from: str,
    split_name: str,
) -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]
]:
    """Build rows for split metrics, overall parse failures, and baseline.

    Returns:
        Tuple of ``(split_group_rows, overall_rows, normalized_rows)``.
    """
    split_group_rows: List[Dict[str, object]] = []
    overall_rows: List[Dict[str, object]] = []
    normalized_rows: List[Dict[str, object]] = []

    for metrics_path in metrics_files:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = infer_model_name(metrics_path, model_name_from)
        rel_path = metrics_path.relative_to(input_dir).as_posix()

        overall = data.get("overall", {})
        overall_total = safe_int(overall.get("total"))
        overall_parse_failures = safe_int(overall.get("parse_failures"))
        overall_pf_rate = None
        if (
            overall_total is not None
            and overall_total > 0
            and overall_parse_failures is not None
        ):
            overall_pf_rate = overall_parse_failures / overall_total

        overall_rows.append(
            {
                "model": model_name,
                "metrics_path": rel_path,
                "overall_total": overall_total,
                "overall_parse_failures": overall_parse_failures,
                "overall_parse_failure_rate": overall_pf_rate,
            }
        )

        split_groups = extract_split_groups(data, split_name=split_name)
        for group_name, scores in split_groups.items():
            total = scores["total"]
            parse_failures = scores["parse_failures"]
            parse_failure_rate = None
            if total is not None and total > 0 and parse_failures is not None:
                parse_failure_rate = parse_failures / total

            row = {
                "model": model_name,
                "metrics_path": rel_path,
                "group_name": group_name,
                "accuracy": scores["accuracy"],
                "parse_failures": parse_failures,
                "total": total,
                "valid": scores["valid"],
                "correct": scores["correct"],
                "parse_failure_rate": parse_failure_rate,
            }
            split_group_rows.append(row)

        normalized_baseline = extract_normalized_baseline(data)
        if normalized_baseline is not None:
            total = normalized_baseline["total"]
            parse_failures = normalized_baseline["parse_failures"]
            parse_failure_rate = None
            if total is not None and total > 0 and parse_failures is not None:
                parse_failure_rate = parse_failures / total

            row = {
                "model": model_name,
                "metrics_path": rel_path,
                "accuracy": normalized_baseline["accuracy"],
                "parse_failures": parse_failures,
                "total": total,
                "valid": normalized_baseline["valid"],
                "correct": normalized_baseline["correct"],
                "parse_failure_rate": parse_failure_rate,
            }
            normalized_rows.append(row)

    return split_group_rows, overall_rows, normalized_rows


def to_wide(
    rows: List[Dict[str, object]],
    value_key: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Convert long rows into wide table by group_name x model."""
    group_names = sorted({str(r["group_name"]) for r in rows})
    models = sorted(
        {str(r["model"]) for r in rows},
        key=model_sort_key,
    )

    grouped: Dict[str, Dict[str, object]] = {}
    for r in rows:
        group = str(r["group_name"])
        model = str(r["model"])
        grouped.setdefault(group, {})[model] = r.get(value_key)

    out_rows: List[Dict[str, object]] = []
    for group in group_names:
        row: Dict[str, object] = {"group_name": group}
        for model in models:
            row[model] = grouped.get(group, {}).get(model)
        out_rows.append(row)

    return ["group_name", *models], out_rows


def write_csv(
    path: Path, rows: List[Dict[str, object]], fields: List[str]
) -> None:
    """Write rows to CSV with fixed field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_groups_by_perturb_family(
    rows: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split groups into type1 and combined type2+3 families."""
    type1_rows: List[Dict[str, object]] = []
    type23_rows: List[Dict[str, object]] = []

    for row in rows:
        group_name = str(row.get("group_name", ""))
        if "| perturb_type1" in group_name:
            type1_rows.append(row)
        elif "| perturb_type2" in group_name or "| perturb_type3" in group_name:
            type23_rows.append(row)

    return type1_rows, type23_rows


def write_split_outputs(
    split_rows: List[Dict[str, object]],
    split_name: str,
    split_output_dir: Path,
    charts: bool,
    overall_rows: List[Dict[str, object]],
    normalized_rows: List[Dict[str, object]],
    include_normalized_in_charts: bool,
) -> List[str]:
    """Write CSV/chart outputs for one split into its own folder."""
    outputs: List[str] = []
    split_output_dir.mkdir(parents=True, exist_ok=True)

    if not split_rows:
        print(
            f"No *_{split_name}.json groups found in per_file_and_type. "
            f"Split folder: {split_output_dir}"
        )
        if charts:
            try:
                create_charts(
                    [],
                    [],
                    overall_rows,
                    normalized_rows,
                    include_normalized_in_charts,
                    split_output_dir,
                )
            except ImportError:
                print(
                    "Could not generate charts: matplotlib is not installed. "
                    "CSV outputs were created successfully."
                )
        return outputs

    type1_rows, type23_rows = split_groups_by_perturb_family(split_rows)

    long_fields = [
        "model",
        "metrics_path",
        "group_name",
        "accuracy",
        "parse_failures",
        "parse_failure_rate",
        "total",
        "valid",
        "correct",
    ]

    if type1_rows:
        write_csv(
            split_output_dir / "perturb_type1_metrics_long.csv",
            sorted(
                type1_rows,
                key=lambda r: (
                    str(r["group_name"]),
                    model_sort_key(str(r["model"])),
                ),
            ),
            long_fields,
        )

        acc_fields, acc_rows = to_wide(type1_rows, "accuracy")
        write_csv(
            split_output_dir / "perturb_type1_accuracy_wide.csv",
            acc_rows,
            acc_fields,
        )

        pf_fields, pf_rows = to_wide(type1_rows, "parse_failures")
        write_csv(
            split_output_dir / "perturb_type1_parse_failures_wide.csv",
            pf_rows,
            pf_fields,
        )
        outputs.extend(
            [
                "perturb_type1_metrics_long.csv",
                "perturb_type1_accuracy_wide.csv",
                "perturb_type1_parse_failures_wide.csv",
            ]
        )

    if type23_rows:
        write_csv(
            split_output_dir / "perturb_type23_metrics_long.csv",
            sorted(
                type23_rows,
                key=lambda r: (
                    str(r["group_name"]),
                    model_sort_key(str(r["model"])),
                ),
            ),
            long_fields,
        )

        acc_fields, acc_rows = to_wide(type23_rows, "accuracy")
        write_csv(
            split_output_dir / "perturb_type23_accuracy_wide.csv",
            acc_rows,
            acc_fields,
        )

        pf_fields, pf_rows = to_wide(type23_rows, "parse_failures")
        write_csv(
            split_output_dir / "perturb_type23_parse_failures_wide.csv",
            pf_rows,
            pf_fields,
        )
        outputs.extend(
            [
                "perturb_type23_metrics_long.csv",
                "perturb_type23_accuracy_wide.csv",
                "perturb_type23_parse_failures_wide.csv",
            ]
        )

    if charts:
        try:
            create_charts(
                type1_rows,
                type23_rows,
                overall_rows,
                normalized_rows,
                include_normalized_in_charts,
                split_output_dir,
            )
        except ImportError:
            print(
                "Could not generate charts: matplotlib is not installed. "
                "CSV outputs were created successfully."
            )

    return outputs


def _label_shortener(group_name: str) -> str:
    """Keep chart y-axis labels readable."""
    parts = group_name.split(" | ")
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return group_name


def _plot_heatmap(
    matrix: List[List[float]],
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    output_path: Path,
    value_fmt: str,
    cmap: str,
) -> None:
    """Create and save a heatmap with text annotations."""
    import matplotlib.pyplot as plt

    if not matrix or not row_labels or not col_labels:
        return

    fig_w = max(8.0, 1.2 * len(col_labels) + 4.0)
    fig_h = max(6.0, 0.5 * len(row_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    img = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(img, ax=ax, fraction=0.035, pad=0.02)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([_label_shortener(r) for r in row_labels])
    ax.set_title(title)

    # Annotate cells for easier numerical comparison.
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            text = "NA" if math.isnan(val) else format(val, value_fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_family_heatmaps(
    rows: List[Dict[str, object]],
    title_prefix: str,
    file_prefix: str,
    charts_dir: Path,
) -> None:
    """Plot accuracy and parse-failure heatmaps for a perturbation family."""
    if not rows:
        return

    models = sorted(
        {str(r["model"]) for r in rows},
        key=model_sort_key,
    )
    groups = sorted({str(r["group_name"]) for r in rows})

    acc_lookup: Dict[Tuple[str, str], float] = {}
    pf_lookup: Dict[Tuple[str, str], float] = {}
    for r in rows:
        group = str(r["group_name"])
        model = str(r["model"])

        acc = r.get("accuracy")
        if isinstance(acc, (int, float)):
            acc_lookup[(group, model)] = float(acc)

        pf = r.get("parse_failures")
        if isinstance(pf, int):
            pf_lookup[(group, model)] = float(pf)

    acc_matrix: List[List[float]] = []
    pf_matrix: List[List[float]] = []
    for group in groups:
        acc_row: List[float] = []
        pf_row: List[float] = []
        for model in models:
            acc_row.append(acc_lookup.get((group, model), float("nan")))
            pf_row.append(pf_lookup.get((group, model), float("nan")))
        acc_matrix.append(acc_row)
        pf_matrix.append(pf_row)

    _plot_heatmap(
        matrix=acc_matrix,
        row_labels=groups,
        col_labels=models,
        title=f"Accuracy: {title_prefix}",
        output_path=charts_dir / f"{file_prefix}_accuracy_heatmap.png",
        value_fmt=".3f",
        cmap="YlGn",
    )

    _plot_heatmap(
        matrix=pf_matrix,
        row_labels=groups,
        col_labels=models,
        title=f"Parse Failures: {title_prefix}",
        output_path=charts_dir / f"{file_prefix}_parse_failures_heatmap.png",
        value_fmt=".0f",
        cmap="OrRd",
    )


def _plot_overall_parse_failures_bar(
    overall_rows: List[Dict[str, object]],
    charts_dir: Path,
) -> None:
    """Plot overall parse-failure bar chart by model."""
    import matplotlib.pyplot as plt

    sorted_overall = sorted(
        overall_rows,
        key=lambda x: model_sort_key(str(x["model"])),
    )
    if not sorted_overall:
        return

    labels = [str(r["model"]) for r in sorted_overall]
    values = [
        safe_int(r.get("overall_parse_failures")) or 0 for r in sorted_overall
    ]

    fig_w = max(8.0, 1.0 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bars = ax.bar(labels, values, color="#d95f02")
    ax.set_title("Overall Parse Failures by Model")
    ax.set_ylabel("Parse failures")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(charts_dir / "overall_parse_failures_bar.png", dpi=150)
    plt.close(fig)


def _plot_normalized_baseline_bars(
    normalized_rows: List[Dict[str, object]],
    charts_dir: Path,
) -> None:
    """Plot accuracy and parse-failure bars for normalized baseline."""
    import matplotlib.pyplot as plt

    sorted_normalized = sorted(
        normalized_rows,
        key=lambda x: model_sort_key(str(x["model"])),
    )
    if not sorted_normalized:
        return

    n_labels = [str(r["model"]) for r in sorted_normalized]
    n_acc = [safe_float_or_nan(r.get("accuracy")) for r in sorted_normalized]
    n_pf = [safe_int(r.get("parse_failures")) or 0 for r in sorted_normalized]

    fig_w = max(8.0, 1.0 * len(n_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bars = ax.bar(n_labels, n_acc, color="#1b9e77")
    ax.set_title("Normalized Baseline Accuracy by Model")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, n_acc):
        if math.isnan(value):
            label = "NA"
            y_pos = 0.0
        else:
            label = f"{value:.3f}"
            y_pos = value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(
        charts_dir / "normalized_baseline_accuracy_bar.png",
        dpi=150,
    )
    plt.close(fig)

    fig_w = max(8.0, 1.0 * len(n_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bars = ax.bar(n_labels, n_pf, color="#7570b3")
    ax.set_title("Normalized Baseline Parse Failures by Model")
    ax.set_ylabel("Parse failures")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, n_pf):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(
        charts_dir / "normalized_baseline_parse_failures_bar.png",
        dpi=150,
    )
    plt.close(fig)


def create_charts(
    type1_rows: List[Dict[str, object]],
    type23_rows: List[Dict[str, object]],
    overall_rows: List[Dict[str, object]],
    normalized_rows: List[Dict[str, object]],
    include_normalized_in_charts: bool,
    output_dir: Path,
) -> None:
    """Generate grouped charts for split comparison outputs.

    This includes perturbation-family heatmaps, a combined heatmap, an overall
    parse-failure bar chart, and optional normalized baseline bar charts.
    """
    charts_dir = output_dir / "charts"

    _plot_family_heatmaps(
        type1_rows,
        title_prefix="Perturb Type 1",
        file_prefix="perturb_type1",
        charts_dir=charts_dir,
    )
    _plot_family_heatmaps(
        type23_rows,
        title_prefix="Perturb Type 2 + 3",
        file_prefix="perturb_type23",
        charts_dir=charts_dir,
    )

    combined_rows = [*type1_rows, *type23_rows]
    _plot_family_heatmaps(
        combined_rows,
        title_prefix="All Perturbation Types (1 + 2 + 3)",
        file_prefix="perturb_all",
        charts_dir=charts_dir,
    )

    _plot_overall_parse_failures_bar(overall_rows, charts_dir)

    if include_normalized_in_charts and normalized_rows:
        _plot_normalized_baseline_bars(normalized_rows, charts_dir)


def main() -> int:
    """Run comparison pipeline and write outputs."""
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    metrics_files = discover_metrics_files(args.input_dir, args.metrics_pattern)
    if not metrics_files:
        raise SystemExit(
            "No metrics.json files found. "
            f"input-dir={args.input_dir}, pattern={args.metrics_pattern}"
        )

    true_group_rows, overall_rows, normalized_rows = build_rows(
        metrics_files=metrics_files,
        input_dir=args.input_dir,
        model_name_from=args.model_name_from,
        split_name="True",
    )
    false_group_rows, _, _ = build_rows(
        metrics_files=metrics_files,
        input_dir=args.input_dir,
        model_name_from=args.model_name_from,
        split_name="False",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_fields = [
        "model",
        "metrics_path",
        "overall_total",
        "overall_parse_failures",
        "overall_parse_failure_rate",
    ]
    write_csv(
        args.output_dir / "overall_parse_failures_summary.csv",
        overall_rows,
        overall_fields,
    )

    if normalized_rows:
        normalized_fields = [
            "model",
            "metrics_path",
            "accuracy",
            "parse_failures",
            "parse_failure_rate",
            "total",
            "valid",
            "correct",
        ]
        write_csv(
            args.output_dir / "normalized_baseline_metrics.csv",
            sorted(
                normalized_rows,
                key=lambda r: model_sort_key(str(r["model"])),
            ),
            normalized_fields,
        )

    true_split_outputs = write_split_outputs(
        split_rows=true_group_rows,
        split_name="True",
        split_output_dir=args.output_dir / "true_split",
        charts=args.charts,
        overall_rows=overall_rows,
        normalized_rows=normalized_rows,
        include_normalized_in_charts=args.include_normalized_in_charts,
    )
    false_split_outputs = write_split_outputs(
        split_rows=false_group_rows,
        split_name="False",
        split_output_dir=args.output_dir / "false_split",
        charts=args.charts,
        overall_rows=overall_rows,
        normalized_rows=normalized_rows,
        include_normalized_in_charts=args.include_normalized_in_charts,
    )

    print(f"Found {len(metrics_files)} metrics files.")
    outputs = [
        "overall_parse_failures_summary.csv",
    ]
    if normalized_rows:
        outputs.append("normalized_baseline_metrics.csv")
    print(f"Wrote: {', '.join(outputs)}")
    print(
        f"Wrote split outputs to: {args.output_dir / 'true_split'} "
        f"({len(true_split_outputs)} files), {args.output_dir / 'false_split'} "
        f"({len(false_split_outputs)} files)"
    )
    if args.charts:
        print(
            "Charts requested: see "
            f"{args.output_dir / 'true_split' / 'charts'} and "
            f"{args.output_dir / 'false_split' / 'charts'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
