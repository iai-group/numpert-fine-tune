"""Generate LaTeX accuracy tables with baseline-model deltas.

This script scans inference metrics under a results directory, extracts
accuracy values from the ``per_file_and_type`` section, filters them to
perturbation groups, and renders a LaTeX table where each non-baseline cell
includes a colored superscript delta against the baseline model for the same
group.

Example terminal usage:
    python -m src.latex.generate_accuracy_delta_table \
        --results-root results/inference/qwen3-8b \
        --output results/inference/qwen3-8b/accuracy_delta_table.tex \
        --baseline-model baseline \
        --perturb-type type1

    python -m src.latex.generate_accuracy_delta_table \
        --results-root results/inference/qwen3-8b \
        --output results/inference/qwen3-8b/type23_accuracy_delta_table.tex \
        --baseline-model baseline \
        --perturb-type type2 \
        --perturb-type type3_masked \
        --perturb-type type3_random

    python -m src.latex.generate_accuracy_delta_table \
        --results-root results/inference/qwen3-8b \
        --output results/inference/qwen3-8b/type23_true_accuracy_delta_table.tex \
        --baseline-model baseline \
        --source-file-pattern "*_True.json" \
        --perturb-type type2 \
        --perturb-type type3_masked \
        --perturb-type type3_random

    python -m src.latex.generate_accuracy_delta_table \
        --results-root models/qwen8-b \
        --output models/qwen8-b/accuracy_delta_table.tex \
        --baseline-model baseline \
        --include-model qwen3_no_approx \
        --group "approximation_True.json | perturb_type1"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SUPPORTED_PERTURB_TYPES = (
    "perturb_type1",
    "perturb_type2",
    "perturb_type3_masked",
    "perturb_type3_random",
)

ORIGINAL_GROUP_NAME = "Original"
ORIGINAL_PERTURBATION_TYPE = "original_normalized"

PERTURBATION_ALIASES = {
    "type1": "perturb_type1",
    "type2": "perturb_type2",
    "type3_masked": "perturb_type3_masked",
    "type3_random": "perturb_type3_random",
    "masked": "perturb_type3_masked",
    "random": "perturb_type3_random",
}

SOURCE_FAMILY_ORDER = {
    "approximation": 0,
    "numeration": 1,
    "range": 2,
    "negative": 3,
    "obfuscation": 4,
}

PERTURBATION_TYPE_ORDER = {
    ORIGINAL_PERTURBATION_TYPE: -1,
    "perturb_type1": 0,
    "perturb_type2": 1,
    "perturb_type3_masked": 2,
    "perturb_type3_random": 3,
}

PERTURBATION_HEADER_LABELS = {
    "perturb_type3_masked": "Masked",
    "perturb_type3_random": "Random",
}


@dataclass(frozen=True)
class GroupScore:
    """Accuracy score for one source-file and perturbation combination."""

    source_file: str
    perturb_type: str
    accuracy: float

    @property
    def group_name(self) -> str:
        return f"{self.source_file} | {self.perturb_type}"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Render a LaTeX table of per-file perturbation accuracies with "
            "deltas against a baseline model."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.latex.generate_accuracy_delta_table \\\n"
            "      --results-root results/inference/qwen3-8b \\\n"
            "      --output results/inference/qwen3-8b/accuracy_delta_table.tex \\\n"
            "      --baseline-model baseline \\\n"
            "      --perturb-type type1\n\n"
            "  python -m src.latex.generate_accuracy_delta_table \\\n"
            "      --results-root results/inference/qwen3-8b \\\n"
            "      --output results/inference/qwen3-8b/type23_accuracy_delta_table.tex \\\n"
            "      --baseline-model baseline \\\n"
            "      --perturb-type type2 \\\n"
            "      --perturb-type type3_masked \\\n"
            "      --perturb-type type3_random\n\n"
            "  python -m src.latex.generate_accuracy_delta_table \\\n"
            "      --results-root results/inference/qwen3-8b \\\n"
            "      --output results/inference/qwen3-8b/type23_true_accuracy_delta_table.tex \\\n"
            "      --baseline-model baseline \\\n"
            "      --source-file-pattern \"*_True.json\" \\\n"
            "      --perturb-type type2 \\\n"
            "      --perturb-type type3_masked \\\n"
            "      --perturb-type type3_random\n\n"
            "  python -m src.latex.generate_accuracy_delta_table \\\n"
            "      --results-root models/qwen8-b \\\n"
            "      --output models/qwen8-b/accuracy_delta_table.tex \\\n"
            "      --baseline-model baseline \\\n"
            "      --include-model qwen3_no_approx \\\n"
            "      --group \"approximation_True.json | perturb_type1\""
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/inference/qwen3-8b"),
        help=(
            "Root folder to search recursively for metrics.json files. "
            "For example: results/inference/qwen3-8b or models/qwen8-b."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .tex file path.",
    )
    parser.add_argument(
        "--baseline-model",
        default="baseline",
        help=(
            "Baseline model folder label. Exact relative folder labels are "
            "supported, and shorthand segment matches like 'baseline' are "
            "resolved when unique."
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Exact group name to include, for example "
            "'approximation_True.json | perturb_type1'. Can be repeated."
        ),
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        help=(
            "Source filename to include, for example 'approximation_True.json'. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--source-file-pattern",
        action="append",
        default=[],
        help=(
            "Glob pattern for source filenames to include, for example "
            "'*_True.json'. Can be repeated."
        ),
    )
    parser.add_argument(
        "--perturb-type",
        action="append",
        default=[],
        help=(
            "Perturbation type to include. Supported values: "
            "perturb_type1, perturb_type2, perturb_type3_masked, "
            "perturb_type3_random. Shorthands like type1 are accepted. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--class-label",
        type=str.lower,
        choices=("true", "false"),
        default=None,
        help=(
            "When set, derive the displayed score from the selected class's "
            "classification report recall instead of overall accuracy."
        ),
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help=(
            "Include the normalized original accuracy as the first column, "
            "labeled 'Original'."
        ),
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help=(
            "Only include these model folder labels in the table. Exact "
            "relative folder labels and unique shorthand segment matches are "
            "supported. Can be repeated. The baseline model is always "
            "included."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for displayed percentage scores and deltas.",
    )
    parser.add_argument(
        "--caption",
        default="",
        help="Optional LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional LaTeX label.",
    )
    return parser


def canonical_perturb_type(raw_value: str) -> str:
    """Normalize perturbation type aliases."""
    lowered = raw_value.strip()
    return PERTURBATION_ALIASES.get(lowered, lowered)


def escape_latex(text: str) -> str:
    """Escape a plain-text string for LaTeX table cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def source_family_label(source_file: str) -> str:
    """Collapse a source filename like approximation_True.json to approximation."""
    stem = Path(source_file).stem
    return re.sub(r"_(true|false)$", "", stem, flags=re.IGNORECASE)


def format_header(group_name: str) -> str:
    """Render a compact header cell for one source family."""
    if group_name == ORIGINAL_GROUP_NAME:
        return ORIGINAL_GROUP_NAME
    source_file, perturb_type = parse_group_name(group_name)
    perturb_type = canonical_perturb_type(perturb_type)
    if perturb_type in PERTURBATION_HEADER_LABELS:
        return escape_latex(PERTURBATION_HEADER_LABELS[perturb_type])
    return escape_latex(source_family_label(source_file).title())


def group_sort_key(group_name: str) -> tuple[int, str, int, str, str]:
    """Sort groups by source family, then perturbation type, then name."""
    if group_name == ORIGINAL_GROUP_NAME:
        return (-1, ORIGINAL_GROUP_NAME, -1, ORIGINAL_GROUP_NAME, ORIGINAL_PERTURBATION_TYPE)
    source_file, perturb_type = parse_group_name(group_name)
    normalized_type = canonical_perturb_type(perturb_type)
    source_family = source_family_label(source_file)
    return (
        SOURCE_FAMILY_ORDER.get(source_family, 999),
        source_family,
        PERTURBATION_TYPE_ORDER.get(normalized_type, 999),
        source_file,
        normalized_type,
    )


def default_caption(groups: Sequence[str]) -> str:
    """Build a placeholder caption that mentions the populated perturbation types."""
    perturb_types = []
    for group_name in groups:
        if group_name == ORIGINAL_GROUP_NAME:
            perturb_type = ORIGINAL_PERTURBATION_TYPE
        else:
            _, perturb_type = parse_group_name(group_name)
        if perturb_type not in perturb_types:
            perturb_types.append(perturb_type)

    perturbation_text = ", ".join(perturb_types)
    return (
        "Placeholder description for the accuracy comparison table populated "
        f"with {perturbation_text}."
    )


def format_score(value: float, digits: int) -> str:
    """Format a score in percentage units."""
    return f"{value * 100:.{digits}f}"


def extract_score(entry: Mapping[str, object], class_label: str | None) -> float:
    """Get the displayed score from a metrics entry.

    When ``class_label`` is set, use the selected class recall from the
    embedded classification report as a class-specific accuracy proxy.
    """
    if class_label is None:
        accuracy = entry.get("accuracy")
        if accuracy is None:
            raise ValueError("Metrics entry is missing an accuracy value")
        return float(accuracy)

    classification_report = entry.get("classification_report")
    if not isinstance(classification_report, Mapping):
        raise ValueError(
            "Metrics entry is missing a classification_report for class scoring"
        )

    class_metrics = classification_report.get(class_label)
    if not isinstance(class_metrics, Mapping):
        raise ValueError(
            f"Metrics entry is missing classification report data for {class_label}"
        )

    class_recall = class_metrics.get("recall")
    if class_recall is None:
        raise ValueError(
            f"Metrics entry is missing recall for class {class_label}"
        )
    return float(class_recall)


def derive_model_family(model_label: str, results_root: Path) -> str:
    """Infer a short model family token such as qwen3 from labels/root."""
    candidates = [Path(model_label).parts[0], results_root.name]
    for candidate in candidates:
        lowered = candidate.lower().lstrip(".")
        if lowered == "baseline":
            continue
        first_chunk = re.split(r"[_/]", lowered)[0]
        first_dash_chunk = first_chunk.split("-")[0]
        if first_dash_chunk:
            return first_dash_chunk
    return results_root.name.lower().replace("-", "_")


def short_model_label(model_label: str, results_root: Path) -> str:
    """Convert a discovered model folder label to a compact display label."""
    top_level = Path(model_label).parts[0].lstrip(".").lower()
    family = derive_model_family(model_label, results_root)

    if top_level == "baseline":
        return f"{family}_baseline"
    if "_no_approx" in top_level:
        return f"{family}_no_approx"
    if "_no_num" in top_level:
        return f"{family}_no_num"
    if "_no_range" in top_level:
        return f"{family}_no_range"
    if "_no_perturb_split" in top_level:
        return f"{family}_no_perturb_split"
    if "normalized_claims" in top_level:
        return f"{family}_normalized_claims"

    cleaned = re.sub(r"[^a-z0-9]+", "_", top_level).strip("_")
    if cleaned.startswith(family + "_"):
        return cleaned
    return f"{family}_{cleaned}" if cleaned else family


def format_delta(
    baseline_value: float,
    value: float,
    digits: int,
    higher_is_better: bool = True,
) -> str:
    """Return a colored superscript delta against the baseline value."""
    delta = (value - baseline_value) * 100
    if abs(delta) < 1e-9:
        return ""
    is_improvement = delta > 0
    color = (
        "numpertdeltaimprove"
        if is_improvement == higher_is_better
        else "numpertdeltadecline"
    )
    sign = "+" if delta > 0 else ""
    return (
        "\\textsuperscript{"
        f"\\textcolor{{{color}}}{{{sign}{delta:.{digits}f}}}"
        "}"
    )


def parse_group_name(group_name: str) -> tuple[str, str]:
    """Split a per-file-and-type group name into source file and type."""
    if " | " not in group_name:
        raise ValueError(f"Unsupported group name format: {group_name}")
    source_file, perturb_type = group_name.split(" | ", maxsplit=1)
    return source_file.strip(), perturb_type.strip()


def load_model_scores(
    metrics_path: Path,
    class_label: str | None = None,
) -> Dict[str, GroupScore]:
    """Load supported per-file perturbation accuracies from one metrics file."""
    with metrics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    scores: Dict[str, GroupScore] = {}
    for key, entry in data.get("per_file_and_type", {}).items():
        group_name = entry.get("group_name", key)
        try:
            source_file, perturb_type = parse_group_name(group_name)
        except ValueError:
            continue

        perturb_type = canonical_perturb_type(perturb_type)
        if perturb_type not in SUPPORTED_PERTURB_TYPES:
            continue

        try:
            score_value = extract_score(entry, class_label)
        except ValueError:
            continue

        score = GroupScore(
            source_file=source_file,
            perturb_type=perturb_type,
            accuracy=score_value,
        )
        scores[score.group_name] = score

    original_entry = data.get("per_perturbation_type", {}).get(
        ORIGINAL_PERTURBATION_TYPE
    )
    if original_entry:
        scores[ORIGINAL_GROUP_NAME] = GroupScore(
            source_file=ORIGINAL_GROUP_NAME,
            perturb_type=ORIGINAL_PERTURBATION_TYPE,
            accuracy=extract_score(original_entry, class_label),
        )

    return scores


def discover_model_metrics(results_root: Path) -> Dict[str, Path]:
    """Find one metrics file per model label below the results root."""
    if not results_root.exists():
        raise ValueError(f"Results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise ValueError(f"Results root is not a directory: {results_root}")

    discovered: Dict[str, Path] = {}
    for metrics_path in sorted(results_root.glob("**/metrics.json")):
        try:
            relative_parent = metrics_path.parent.relative_to(results_root)
        except ValueError as exc:
            raise ValueError(
                f"Metrics path {metrics_path} is not under {results_root}"
            ) from exc

        model_label = relative_parent.as_posix()
        if model_label == ".":
            model_label = results_root.name
        if model_label in discovered:
            raise ValueError(
                "Found multiple metrics files for model label "
                f"'{model_label}'. Please make the results root more specific "
                "or remove duplicate metrics files."
            )
        discovered[model_label] = metrics_path

    if not discovered:
        raise ValueError(f"No metrics.json files found under {results_root}")
    return discovered


def select_models(
    discovered: Mapping[str, Path],
    baseline_model: str,
    include_models: Sequence[str],
) -> tuple[Dict[str, Path], str]:
    """Filter the discovered models while keeping the baseline available."""
    resolved_baseline = resolve_model_label(discovered, baseline_model)

    if not include_models:
        return dict(discovered), resolved_baseline

    selected_labels = {resolved_baseline}
    for requested_label in include_models:
        selected_labels.add(resolve_model_label(discovered, requested_label))

    return {
        label: path
        for label, path in discovered.items()
        if label in selected_labels
    }, resolved_baseline


def resolve_model_label(
    discovered: Mapping[str, Path],
    requested_label: str,
) -> str:
    """Resolve a model label from an exact or unique folder-segment match."""
    if requested_label in discovered:
        return requested_label

    suffix_matches = [
        label
        for label in discovered
        if label.endswith("/" + requested_label)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"Model label '{requested_label}' is ambiguous. Matches: "
            + ", ".join(sorted(suffix_matches))
        )

    segment_matches = [
        label
        for label in discovered
        if requested_label in Path(label).parts
    ]
    if len(segment_matches) == 1:
        return segment_matches[0]
    if len(segment_matches) > 1:
        raise ValueError(
            f"Model label '{requested_label}' is ambiguous. Matches: "
            + ", ".join(sorted(segment_matches))
        )

    available = ", ".join(sorted(discovered))
    raise ValueError(
        f"Model label '{requested_label}' not found. Available labels: {available}"
    )


def derive_selected_groups(
    baseline_scores: Mapping[str, GroupScore],
    exact_groups: Sequence[str],
    source_files: Sequence[str],
    source_file_patterns: Sequence[str],
    perturb_types: Sequence[str],
    include_original: bool,
) -> List[str]:
    """Choose which group columns to render."""
    normalized_types = {
        canonical_perturb_type(value) for value in perturb_types if value.strip()
    }
    exact_source_files = set(source_files)
    source_patterns = [pattern for pattern in source_file_patterns if pattern]
    unsupported_types = normalized_types.difference(SUPPORTED_PERTURB_TYPES)
    if unsupported_types:
        supported = ", ".join(SUPPORTED_PERTURB_TYPES)
        raise ValueError(
            "Unsupported perturb types requested: "
            f"{', '.join(sorted(unsupported_types))}. Supported: {supported}"
        )

    selected: List[str] = []

    if exact_groups:
        for group_name in exact_groups:
            source_file, perturb_type = parse_group_name(group_name)
            perturb_type = canonical_perturb_type(perturb_type)
            normalized_group = f"{source_file} | {perturb_type}"
            if normalized_group not in baseline_scores:
                raise ValueError(
                    "Group not found in baseline metrics or not a supported "
                    f"perturbation group: {normalized_group}"
                )
            selected.append(normalized_group)

    if source_files or source_patterns or normalized_types:
        for group_name, score in baseline_scores.items():
            if group_name == ORIGINAL_GROUP_NAME:
                continue
            matches_source = (
                (
                    not exact_source_files
                    or score.source_file in exact_source_files
                )
                and (
                    not source_patterns
                    or any(
                        fnmatch.fnmatch(score.source_file, pattern)
                        for pattern in source_patterns
                    )
                )
            )
            matches_type = (
                not normalized_types or score.perturb_type in normalized_types
            )
            if matches_source and matches_type and group_name not in selected:
                selected.append(group_name)

    if not selected:
        selected = [
            group_name
            for group_name in baseline_scores
            if group_name != ORIGINAL_GROUP_NAME
        ]

    selected = sorted(selected, key=group_sort_key)

    if include_original:
        if ORIGINAL_GROUP_NAME not in baseline_scores:
            raise ValueError(
                "Baseline metrics do not contain an original_normalized accuracy"
            )
        selected = [
            ORIGINAL_GROUP_NAME,
            *[group for group in selected if group != ORIGINAL_GROUP_NAME],
        ]

    return selected


def compute_best_scores(
    model_scores: Mapping[str, Mapping[str, GroupScore]],
    groups: Sequence[str],
) -> Dict[str, float]:
    """Find the best accuracy value per selected group."""
    best_scores: Dict[str, float] = {}
    for group_name in groups:
        available = [
            scores[group_name].accuracy
            for scores in model_scores.values()
            if group_name in scores
        ]
        if not available:
            raise ValueError(f"No scores available for selected group {group_name}")
        best_scores[group_name] = max(available)
    return best_scores


def render_table(
    model_scores: Mapping[str, Mapping[str, GroupScore]],
    groups: Sequence[str],
    baseline_model: str,
    results_root: Path,
    digits: int,
    caption: str = "",
    label: str = "",
) -> str:
    """Render the LaTeX table."""
    baseline_scores = model_scores[baseline_model]
    best_scores = compute_best_scores(model_scores, groups)
    caption_text = caption or default_caption(groups)

    ordered_models = [baseline_model] + sorted(
        label_ for label_ in model_scores if label_ != baseline_model
    )

    lines = [
        "% Requires \\usepackage[table]{xcolor}",
        "\\begin{table*}[ht]",
        "\\centering",
        "\\definecolor{numpertdeltaimprove}{RGB}{0,100,0}",
        "\\definecolor{numpertdeltadecline}{RGB}{180,35,24}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(groups) + "}",
        "\\hline",
        "Model & " + " & ".join(format_header(group) for group in groups)
        + " " + "\\\\",
        "\\hline",
    ]

    for model_label in ordered_models:
        row_cells = [escape_latex(short_model_label(model_label, results_root))]
        scores_for_model = model_scores[model_label]
        for group_name in groups:
            score = scores_for_model.get(group_name)
            if score is None:
                row_cells.append("--")
                continue

            cell = format_score(score.accuracy, digits)
            if abs(score.accuracy - best_scores[group_name]) < 1e-9:
                cell = f"\\textbf{{{cell}}}"

            if model_label != baseline_model:
                cell += format_delta(
                    baseline_scores[group_name].accuracy,
                    score.accuracy,
                    digits,
                )

            row_cells.append(cell)

        lines.append(" & ".join(row_cells) + " " + "\\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}",
    ])

    lines.append(f"\\caption{{{escape_latex(caption_text)}}}")
    if label:
        lines.append(f"\\label{{{escape_latex(label)}}}")

    lines.append("\\end{table*}")
    return "\n".join(lines)


def ensure_baseline_groups(
    model_scores: Mapping[str, Mapping[str, GroupScore]],
    baseline_model: str,
    groups: Iterable[str],
) -> None:
    """Ensure every selected group exists in the baseline model metrics."""
    baseline_groups = model_scores[baseline_model]
    missing = [group for group in groups if group not in baseline_groups]
    if missing:
        raise ValueError(
            "Selected groups missing from baseline metrics: "
            + ", ".join(missing)
        )


def write_output(output_path: Path, latex: str) -> None:
    """Write the generated LaTeX to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    discovered = discover_model_metrics(args.results_root)
    selected_models, baseline_model = select_models(
        discovered,
        baseline_model=args.baseline_model,
        include_models=args.include_model,
    )

    model_scores = {
        model_label: load_model_scores(metrics_path, class_label=args.class_label)
        for model_label, metrics_path in selected_models.items()
    }

    baseline_scores = model_scores[baseline_model]
    groups = derive_selected_groups(
        baseline_scores,
        exact_groups=args.group,
        source_files=args.source_file,
        source_file_patterns=args.source_file_pattern,
        perturb_types=args.perturb_type,
        include_original=args.include_original,
    )
    ensure_baseline_groups(model_scores, baseline_model, groups)

    latex = render_table(
        model_scores,
        groups,
        baseline_model=baseline_model,
        results_root=args.results_root,
        digits=args.digits,
        caption=args.caption,
        label=args.label,
    )
    write_output(args.output, latex)
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()