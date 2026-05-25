"""Generate LaTeX accuracy tables with baseline-model deltas.

Scans inference metrics under a results directory, extracts accuracy values
from the ``per_file_and_type`` section, and renders a LaTeX table where each
non-baseline cell includes a colored superscript delta against the baseline.
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

_ONLY_CATEGORY_ALIASES = (
    ("only_obfuscation_masked", "masked"),
    ("only_obfuscation_random", "random"),
    ("only_approximation", "approx"),
    ("only_numeration", "num"),
    ("only_range", "range"),
    ("only_negative", "negation"),
)

MODEL_SUBSCRIPTS: Dict[str, str] = {
    "baseline": "Baseline",
    "no_approx": "Excl-Approx",
    "no_num": "Excl-Num",
    "no_range": "Excl-Range",
    "no_perturb_split": "Excl-Split",
    "normalized_claims": "Norm-Claims",
}


@dataclass(frozen=True)
class GroupScore:
    """Accuracy score for one source-file and perturbation combination."""

    source_file: str
    perturb_type: str
    accuracy: float

    @property
    def group_name(self) -> str:
        """Return a combined key joining source file and perturbation type."""
        return f"{self.source_file} | {self.perturb_type}"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Render a LaTeX table of per-file perturbation accuracies with "
            "deltas against a baseline model."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/inference/qwen3-8b"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-model", default="baseline")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--source-file-pattern", action="append", default=[])
    parser.add_argument("--perturb-type", action="append", default=[])
    parser.add_argument(
        "--class-label",
        type=str.lower,
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--include-model", action="append", default=[])
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--caption", default="")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--model-name-map",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file mapping short model labels to LaTeX commands. "
            'Each line: {"short_label": "...", "command": "\\\\cmd", '
            '"definition": "$...$"}. When provided, table cells use the '
            "command (e.g. \\\\mbase) instead of the full math notation."
        ),
    )
    return parser


def canonical_perturb_type(raw_value: str) -> str:
    """Resolve a perturbation type alias to its canonical name."""
    lowered = raw_value.strip()
    return PERTURBATION_ALIASES.get(lowered, lowered)


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
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
    """Strip the boolean suffix from a source filename, returning the family.

    For example, ``approximation_True.json`` becomes ``approximation``.
    """
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
        return (
            -1,
            ORIGINAL_GROUP_NAME,
            -1,
            ORIGINAL_GROUP_NAME,
            ORIGINAL_PERTURBATION_TYPE,
        )
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
    """Build a placeholder caption mentioning the populated perturbation types.

    Lists each distinct perturbation type found across the given groups.
    """
    seen: list[str] = []
    for group_name in groups:
        perturb_type = (
            ORIGINAL_PERTURBATION_TYPE
            if group_name == ORIGINAL_GROUP_NAME
            else parse_group_name(group_name)[1]
        )
        if perturb_type not in seen:
            seen.append(perturb_type)
    return (
        "Placeholder description for the accuracy comparison table populated "
        f"with {', '.join(seen)}."
    )


def format_score(value: float, digits: int) -> str:
    """Format accuracy as a percentage string with the given precision."""
    return f"{value * 100:.{digits}f}"


def extract_score(
    entry: Mapping[str, object], class_label: str | None
) -> float:
    """Get the displayed score from a metrics entry.

    When ``class_label`` is set, use the selected class recall from the
    embedded classification report as a class-specific accuracy proxy.
    """
    if class_label is None:
        accuracy = entry.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            raise ValueError("Metrics entry is missing an accuracy value")
        return float(accuracy)

    classification_report = entry.get("classification_report")
    if not isinstance(classification_report, Mapping):
        # Older metrics files lack classification_report; fall back to accuracy.
        accuracy = entry.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            raise ValueError(
                "Metrics entry is missing a classification_report for class scoring"  # noqa: E501
            )
        return float(accuracy)

    class_metrics = classification_report.get(class_label)
    if not isinstance(class_metrics, Mapping):
        raise ValueError(
            f"Metrics entry is missing classification report data for "
            f"{class_label}"
        )

    class_recall = class_metrics.get("recall")
    if class_recall is None:
        raise ValueError(
            f"Metrics entry is missing recall for class {class_label}"
        )
    return float(class_recall)


def short_model_label(  # noqa: C901
    model_label: str, results_root: Path
) -> str:
    """Convert a discovered model folder label to a compact display label."""
    top = Path(model_label).parts[0].lstrip(".").lower()

    family = next(
        (
            re.split(r"[_\-/]", candidate.lower().lstrip("."))[0]
            for candidate in (top, results_root.name.lower().lstrip("."))
            if re.split(r"[_\-/]", candidate.lower().lstrip("."))[0]
            and re.split(r"[_\-/]", candidate.lower().lstrip("."))[0]
            != "baseline"
        ),
        results_root.name.lower().replace("-", "_"),
    )

    if top == "baseline":
        return f"{family}_baseline"
    for tag in ("_no_approx", "_no_num", "_no_range", "_no_perturb_split"):
        if tag in top:
            return f"{family}{tag}"
    if "normalized_claims" in top:
        return f"{family}_normalized_claims"
    for pattern, alias in _ONLY_CATEGORY_ALIASES:
        if pattern in top:
            return f"{family}_{alias}"
    if "all_perturb" in top:
        return f"{family}_all_perturb"
    if "full_data" in top:
        return f"{family}_full_data"
    if "base" in top:
        if "two" in top:
            return f"{family}_base_two_shot"
        if "pap" in top:
            return f"{family}_base_pap"

    cleaned = re.sub(r"[^a-z0-9]+", "_", top).strip("_")
    if cleaned.startswith(family + "_"):
        return cleaned
    return f"{family}_{cleaned}" if cleaned else family


def formal_model_label(model_label: str, results_root: Path) -> str:
    r"""Return the formal LaTeX math notation for a model.

    For example: $\text{M}_{\text{Excl-Range}}$.
    """
    short = short_model_label(model_label, results_root)
    idx = short.find("_")
    suffix = short[idx + 1 :] if idx != -1 else short
    subscript = MODEL_SUBSCRIPTS.get(suffix, suffix.replace("_", "-").title())
    return f"$\\text{{M}}_{{\\text{{{subscript}}}}}$"


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


def load_model_name_map(path: Path) -> Dict[str, Dict[str, str]]:
    """Load JSONL model name map keyed by short_label.

    Each line must have ``short_label``, ``command``, and ``definition`` fields.
    """
    entries: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["short_label"]] = entry
    return entries


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


def resolve_model_label(
    discovered: Mapping[str, Path],
    requested_label: str,
) -> str:
    """Resolve a model label from an exact or unique folder-segment match."""
    if requested_label in discovered:
        return requested_label

    for candidates in (
        [
            label
            for label in discovered
            if label.endswith("/" + requested_label)
        ],
        [label for label in discovered if requested_label in Path(label).parts],
    ):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Model label '{requested_label}' is ambiguous. Matches: "
                + ", ".join(sorted(candidates))
            )

    available = ", ".join(sorted(discovered))
    raise ValueError(
        f"Model label '{requested_label}' not found. "
        f"Available labels: {available}"
    )


def select_models(
    discovered: Mapping[str, Path],
    baseline_model: str,
    include_models: Sequence[str],
) -> tuple[Dict[str, Path], str]:
    """Filter the discovered models while keeping the baseline available."""
    resolved_baseline = resolve_model_label(discovered, baseline_model)

    if not include_models:
        return dict(discovered), resolved_baseline

    keep = {resolved_baseline} | {
        resolve_model_label(discovered, m) for m in include_models
    }
    return {
        label: path for label, path in discovered.items() if label in keep
    }, resolved_baseline


def _matches_group_filters(
    score: GroupScore,
    exact_source_files: set[str],
    source_patterns: list[str],
    normalized_types: set[str],
) -> bool:
    """Check if a group score matches the provided filter criteria."""
    if exact_source_files and score.source_file not in exact_source_files:
        return False
    if source_patterns and not any(
        fnmatch.fnmatch(score.source_file, p) for p in source_patterns
    ):
        return False
    if normalized_types and score.perturb_type not in normalized_types:
        return False
    return True


def derive_selected_groups(  # noqa: C901
    baseline_scores: Mapping[str, GroupScore],
    exact_groups: Sequence[str],
    source_files: Sequence[str],
    source_file_patterns: Sequence[str],
    perturb_types: Sequence[str],
    include_original: bool,
) -> List[str]:
    """Choose which group columns to render."""
    normalized_types = {
        canonical_perturb_type(value)
        for value in perturb_types
        if value.strip()
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
            if _matches_group_filters(
                score, exact_source_files, source_patterns, normalized_types
            ):
                if group_name not in selected:
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
                "Baseline metrics do not contain an "
                "original_normalized accuracy"
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
            raise ValueError(
                f"No scores available for selected group {group_name}"
            )
        best_scores[group_name] = max(available)
    return best_scores


def render_table(  # noqa: C901
    model_scores: Mapping[str, Mapping[str, GroupScore]],
    groups: Sequence[str],
    baseline_model: str,
    results_root: Path,
    digits: int,
    caption: str = "",
    label: str = "",
    name_map: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Render the LaTeX table."""
    best_scores = compute_best_scores(model_scores, groups)
    caption_text = caption or default_caption(groups)

    def _model_slot(lbl: str) -> int:
        short = short_model_label(lbl, results_root)
        if "_base_two_shot" in short:
            return 0
        if "_base_pap" in short:
            return 1
        if "_no_perturb_split" in short:
            return 2
        return 3

    others = [lbl for lbl in model_scores if lbl != baseline_model]
    ordered_models = [baseline_model] + sorted(
        others, key=lambda lbl: (_model_slot(lbl), lbl)
    )

    lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage[table]{xcolor}",
        "\\begin{table*}[ht]",
        "\\centering",
        "\\definecolor{numpertdeltaimprove}{RGB}{0,100,0}",
        "\\definecolor{numpertdeltadecline}{RGB}{180,35,24}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(groups) + "}",
        "\\toprule",
        "\\textbf{Model} & "
        + " & ".join(f"\\textbf{{{format_header(g)}}}" for g in groups)
        + " \\\\",
        "\\midrule",
    ]

    for model_label in ordered_models:
        short = short_model_label(model_label, results_root)
        if name_map and short in name_map:
            model_cell = name_map[short]["command"]
        else:
            model_cell = formal_model_label(model_label, results_root)
        row_cells = [model_cell]
        scores_for_model = model_scores[model_label]
        own_original = scores_for_model.get(ORIGINAL_GROUP_NAME)
        for group_name in groups:
            score = scores_for_model.get(group_name)
            if score is None:
                row_cells.append("--")
                continue

            cell = format_score(score.accuracy, digits)
            if abs(score.accuracy - best_scores[group_name]) < 1e-9:
                cell = f"\\textbf{{{cell}}}"

            if group_name != ORIGINAL_GROUP_NAME and own_original is not None:
                cell += format_delta(
                    own_original.accuracy,
                    score.accuracy,
                    digits,
                )

            row_cells.append(cell)

        lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

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
    """Write LaTeX to a file, creating parent directories as needed."""
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
        model_label: load_model_scores(
            metrics_path, class_label=args.class_label
        )
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

    name_map = (
        load_model_name_map(args.model_name_map)
        if args.model_name_map
        else None
    )
    latex = render_table(
        model_scores,
        groups,
        baseline_model=baseline_model,
        results_root=args.results_root,
        digits=args.digits,
        caption=args.caption,
        label=args.label,
        name_map=name_map,
    )
    write_output(args.output, latex)
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
