"""Generate LaTeX accuracy tables with baseline-model deltas.

Scans inference metrics under a results directory, extracts accuracy values
from the ``per_file_and_type`` section, and renders a LaTeX table where each
non-baseline cell includes a colored superscript delta against the baseline.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
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

# Compact header variants used when --abbreviate-headers is set. Source
# families absent from the map keep their default title-cased form (e.g.
# "Range" stays "Range" because no shorter form is meaningful).
SOURCE_FAMILY_ABBREVIATIONS = {
    "approximation": "Approx.",
    "numeration": "Num.",
    "negative": "Neg.",
    "obfuscation": "Obfusc.",
}

PERTURBATION_HEADER_ABBREVIATIONS = {
    "perturb_type3_masked": "Mask.",
    "perturb_type3_random": "Rand.",
}

ORIGINAL_GROUP_ABBREVIATION = "Orig."

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

# Fallback ordering for non-baseline rows when no name_map is supplied.
# When a model-name-map JSONL is provided, its line order takes precedence —
# this tuple is only consulted for short labels not present in that map.
# Short labels not listed here fall after the listed ones, sorted by label.
MODEL_ORDER: tuple[str, ...] = (
    # Group 1 — base
    "qwen3_baseline",
    "qwen3_base_two_shot",
    "qwen3_base_pap",
    "qwen3_32b_baseline",
    "qwen3_no_perturb_split",
    # Group 2 — allperturb. qwen3_pt_full_data is reserved for the
    # \mptfulldata model added separately; the slot keeps it positioned
    # ahead of the all-perturb size sweep when its name_map entry lands.
    "qwen3_pt_full_data",
    "qwen3_0_6b_all_perturb",
    "qwen3_1_7b_all_perturb",
    "qwen3_4b_all_perturb",
    "qwen3_all_perturb",
    # Group 3 — only
    "qwen3_approx",
    "qwen3_num",
    "qwen3_range",
    "qwen3_negation",
    "qwen3_masked",
    "qwen3_random",
    # Group 4 — excl. qwen3_only_perturbs (rendered as \mnooriginal)
    # leads the excluded-data group as the "no original" variant.
    "qwen3_only_perturbs",
    "qwen3_no_approx",
    "qwen3_no_num",
    "qwen3_no_range",
    "qwen3_no_negation",
    "qwen3_no_masked",
    "qwen3_no_random",
)


@dataclass(frozen=True)
class GroupScore:
    """Accuracy score for one source-file and perturbation combination."""

    source_file: str
    perturb_type: str
    accuracy: float
    parse_failures: int | None = None
    total: int | None = None

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
    parser.add_argument(
        "--baseline-model",
        default=None,
        help=(
            "Optional discovered-folder label to render as the first row. "
            "When omitted, all rows are treated equally; deltas remain "
            "row-self-relative against the row's own Original cell."
        ),
    )
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
    parser.add_argument(
        "--original-class-label",
        type=str.lower,
        choices=("true", "false"),
        default=None,
        help=(
            "Class label used for the Original column only. Defaults to "
            "--class-label. Useful for flipped tables where Original "
            "should reflect the source-file label (e.g. 'true' for "
            "true-split-flipped) while perturbed cells score against the "
            "flipped class."
        ),
    )
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument(
        "--abbreviate-headers",
        action="store_true",
        help=(
            "Use shortened column headers (Approx., Num., Neg., Obfusc., "
            "Mask., Rand., Orig.) to make wide tables more compact. Range "
            "stays unchanged."
        ),
    )
    parser.add_argument(
        "--group-separators",
        action="store_true",
        help=(
            "Add a vertical rule after the model column (l|ccc...). "
            "Midrules and rotated row-group labels between the four model "
            "groups (base / all-perturb / only-X / excl-X) are always "
            "emitted."
        ),
    )
    parser.add_argument(
        "--macro-average-column",
        action="store_true",
        help=(
            "Add a per-row macro-averaged accuracy column. The average is the "
            "unweighted mean of the displayed perturbation cells and "
            "excludes the Original column."
        ),
    )
    parser.add_argument(
        "--micro-average-column",
        action="store_true",
        help=(
            "Add a per-row micro-averaged accuracy column. Each displayed "
            "perturbation cell is weighted by its sample count (the "
            "``total`` field from metrics). The Original column is excluded."
        ),
    )
    parser.add_argument(
        "--parse-failures-display",
        type=str.lower,
        choices=("column", "caption"),
        default=None,
        help=(
            "Show each model's parse-failure rate, macro-averaged across "
            "the rendered groups. 'column' adds a rightmost column; "
            "'caption' appends a per-model breakdown to the caption. Omit "
            "to hide parse-failure information."
        ),
    )
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
    parser.add_argument(
        "--model-dir-map",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file mapping discovered model directory names to "
            'short labels. Each line: {"dir": "<dir-name>", '
            '"short_label": "..."}. When provided, listed directories use the '
            "explicit short label instead of the heuristic derived from the "
            "results-root name. Unlisted directories fall back to the "
            "heuristic."
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


def format_header(group_name: str, abbreviate: bool = False) -> str:
    """Render a compact header cell for one source family.

    When ``abbreviate`` is true, headers use shortened forms (``Approx.``,
    ``Num.``, ``Mask.``, ``Orig.``, etc.) to keep wide tables narrow.
    """
    if group_name == ORIGINAL_GROUP_NAME:
        return (
            ORIGINAL_GROUP_ABBREVIATION if abbreviate else ORIGINAL_GROUP_NAME
        )
    source_file, perturb_type = parse_group_name(group_name)
    perturb_type = canonical_perturb_type(perturb_type)
    if perturb_type in PERTURBATION_HEADER_LABELS:
        label_map = (
            PERTURBATION_HEADER_ABBREVIATIONS
            if abbreviate
            else PERTURBATION_HEADER_LABELS
        )
        return escape_latex(label_map[perturb_type])
    family = source_family_label(source_file)
    if abbreviate and family in SOURCE_FAMILY_ABBREVIATIONS:
        return escape_latex(SOURCE_FAMILY_ABBREVIATIONS[family])
    return escape_latex(family.title())


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
        raise ValueError(
            "Metrics entry is missing a classification_report for class scoring"
        )

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
    model_label: str,
    results_root: Path,
    dir_map: Mapping[str, str] | None = None,
) -> str:
    """Convert a discovered model folder label to a compact display label."""
    if dir_map:
        if model_label in dir_map:
            return dir_map[model_label]
        leaf = Path(model_label).name
        if leaf in dir_map:
            return dir_map[leaf]
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
    if "base" in top:
        if "two" in top:
            return f"{family}_base_two_shot"

    cleaned = re.sub(r"[^a-z0-9]+", "_", top).strip("_")
    if cleaned.startswith(family + "_"):
        return cleaned
    return f"{family}_{cleaned}" if cleaned else family


def formal_model_label(
    model_label: str,
    results_root: Path,
    dir_map: Mapping[str, str] | None = None,
) -> str:
    r"""Return the formal LaTeX math notation for a model.

    For example: $\text{M}_{\text{Excl-Range}}$.
    """
    short = short_model_label(model_label, results_root, dir_map)
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


_ONLY_X_SHORT_LABELS = frozenset(
    {
        "qwen3_approx",
        "qwen3_num",
        "qwen3_range",
        "qwen3_negation",
        "qwen3_random",
        "qwen3_masked",
    }
)


def _model_group(short_label: str) -> str:
    """Classify a model into a row-group for \\midrule placement.

    Returns one of: ``base``, ``allperturb``, ``only``, ``excl``.
    """
    if short_label == "qwen3_no_perturb_split":
        return "base"
    if short_label == "qwen3_only_perturbs":
        return "excl"
    if short_label.startswith("qwen3_no_"):
        return "excl"
    if (
        short_label.endswith("_all_perturb")
        or short_label == "qwen3_pt_full_data"
    ):
        return "allperturb"
    if short_label in _ONLY_X_SHORT_LABELS:
        return "only"
    return "base"


ROW_GROUP_LABELS: Dict[str, str] = {
    "base": "\\grouponetable",
    "allperturb": "\\grouptwotable",
    "only": "\\groupthreetable",
    "excl": "\\groupfourtable",
}


def _format_average_cell(
    value: float | None,
    best: float | None,
    baseline_value: float | None,
    digits: int,
) -> str:
    """Format a macro/micro average cell with optional bold + delta."""
    if value is None:
        return "--"
    cell = format_score(value, digits)
    if best is not None and abs(value - best) < 1e-9:
        cell = f"\\textbf{{{cell}}}"
    if baseline_value is not None:
        cell += format_delta(baseline_value, value, digits)
    return cell


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


def load_model_dir_map(path: Path) -> Dict[str, str]:
    """Load JSONL directory-name to short-label map.

    Each line must have ``dir`` and ``short_label`` fields.
    """
    entries: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["dir"]] = entry["short_label"]
    return entries


def load_model_scores(
    metrics_path: Path,
    class_label: str | None = None,
    original_class_label: str | None = None,
) -> Dict[str, GroupScore]:
    """Load supported per-file perturbation accuracies from one metrics file.

    ``original_class_label`` overrides ``class_label`` for the Original
    column only — useful when the perturbed cells score against the
    flipped class while Original should keep the source-file's own class.
    """
    effective_original_label = (
        original_class_label
        if original_class_label is not None
        else class_label
    )
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
            parse_failures=_extract_int(entry, "parse_failures"),
            total=_extract_int(entry, "total"),
        )
        scores[score.group_name] = score

    original_entry = data.get("per_perturbation_type", {}).get(
        ORIGINAL_PERTURBATION_TYPE
    )
    if original_entry:
        scores[ORIGINAL_GROUP_NAME] = GroupScore(
            source_file=ORIGINAL_GROUP_NAME,
            perturb_type=ORIGINAL_PERTURBATION_TYPE,
            accuracy=extract_score(original_entry, effective_original_label),
            parse_failures=_extract_int(original_entry, "parse_failures"),
            total=_extract_int(original_entry, "total"),
        )

    return scores


def _extract_int(entry: Mapping[str, object], key: str) -> int | None:
    """Return an integer-valued field from a metrics entry, if present."""
    value = entry.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def discover_model_metrics(results_root: Path) -> Dict[str, Path]:
    """Find one metrics file per model label below the results root."""
    if not results_root.exists():
        raise ValueError(f"Results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise ValueError(f"Results root is not a directory: {results_root}")

    collected: List[Path] = []
    for dirpath, _dirs, files in os.walk(results_root, followlinks=True):
        if "metrics.json" in files:
            collected.append(Path(dirpath) / "metrics.json")

    discovered: Dict[str, Path] = {}
    for metrics_path in sorted(collected):
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
    baseline_model: str | None,
    include_models: Sequence[str],
) -> tuple[Dict[str, Path], str | None]:
    """Filter the discovered models while keeping the baseline available.

    ``baseline_model`` is optional: when ``None``, no row is singled out and
    all discovered models flow through normally.
    """
    resolved_baseline = (
        resolve_model_label(discovered, baseline_model)
        if baseline_model is not None
        else None
    )

    if not include_models:
        return dict(discovered), resolved_baseline

    keep = {resolve_model_label(discovered, m) for m in include_models}
    if resolved_baseline is not None:
        keep.add(resolved_baseline)
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


def compute_parse_failure_rate(
    scores: Mapping[str, GroupScore],
    groups: Sequence[str],
) -> float | None:
    """Macro-average parse-failure rate (%) across the given groups.

    Each rendered group's rate (``parse_failures / total * 100``) contributes
    equally to the mean. Groups missing either field, or with a zero total,
    are skipped. Returns ``None`` when no group has both fields recorded.
    """
    rates: list[float] = []
    for group_name in groups:
        score = scores.get(group_name)
        if score is None:
            continue
        if score.parse_failures is None or score.total is None:
            continue
        if score.total == 0:
            continue
        rates.append(score.parse_failures / score.total * 100)
    if not rates:
        return None
    return sum(rates) / len(rates)


def compute_macro_average(
    scores: Mapping[str, GroupScore],
    groups: Sequence[str],
) -> float | None:
    """Unweighted mean of accuracies across non-original selected groups."""
    values = [
        scores[group_name].accuracy
        for group_name in groups
        if group_name != ORIGINAL_GROUP_NAME and group_name in scores
    ]
    if not values:
        return None
    return sum(values) / len(values)


def compute_micro_average(
    scores: Mapping[str, GroupScore],
    groups: Sequence[str],
) -> float | None:
    """Sample-weighted mean of accuracies across non-original groups.

    Each group's accuracy is weighted by its ``total`` field. Groups missing
    a usable total are skipped; the result is ``None`` only when every
    non-original group is unusable.
    """
    weighted_sum = 0.0
    total_sum = 0
    for group_name in groups:
        if group_name == ORIGINAL_GROUP_NAME:
            continue
        score = scores.get(group_name)
        if score is None or score.total is None or score.total <= 0:
            continue
        weighted_sum += score.accuracy * score.total
        total_sum += score.total
    if total_sum == 0:
        return None
    return weighted_sum / total_sum


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
    baseline_model: str | None,
    results_root: Path,
    digits: int,
    caption: str = "",
    label: str = "",
    name_map: Mapping[str, Mapping[str, str]] | None = None,
    dir_map: Mapping[str, str] | None = None,
    show_parse_failures: bool = False,
    abbreviate_headers: bool = False,
    parse_failures_display: str | None = None,
    macro_average_column: bool = False,
    micro_average_column: bool = False,
    group_separators: bool = False,
) -> str:
    """Render the LaTeX table."""
    best_scores = compute_best_scores(model_scores, groups)
    caption_text = caption or default_caption(groups)

    name_map_order = list(name_map.keys()) if name_map else []

    def _model_slot(lbl: str) -> int:
        short = short_model_label(lbl, results_root, dir_map)
        if short in name_map_order:
            return name_map_order.index(short)
        try:
            return len(name_map_order) + MODEL_ORDER.index(short)
        except ValueError:
            return len(name_map_order) + len(MODEL_ORDER)

    others = [lbl for lbl in model_scores if lbl != baseline_model]
    sorted_others = sorted(others, key=lambda lbl: (_model_slot(lbl), lbl))
    ordered_models = (
        [baseline_model, *sorted_others]
        if baseline_model is not None
        else sorted_others
    )

    def _row_label(model_label: str) -> str:
        short = short_model_label(model_label, results_root, dir_map)
        if name_map and short in name_map:
            return name_map[short]["command"]
        return formal_model_label(model_label, results_root, dir_map)

    show_rate_column = parse_failures_display == "column"
    rate_by_model: Dict[str, float | None] = {}
    if parse_failures_display in ("column", "caption"):
        rate_by_model = {
            model_label: compute_parse_failure_rate(
                model_scores[model_label], groups
            )
            for model_label in ordered_models
        }

    macro_by_model: Dict[str, float | None] = {}
    if macro_average_column:
        macro_by_model = {
            model_label: compute_macro_average(
                model_scores[model_label], groups
            )
            for model_label in ordered_models
        }
    micro_by_model: Dict[str, float | None] = {}
    if micro_average_column:
        micro_by_model = {
            model_label: compute_micro_average(
                model_scores[model_label], groups
            )
            for model_label in ordered_models
        }
    best_macro = (
        max(
            (v for v in macro_by_model.values() if v is not None),
            default=None,
        )
        if macro_average_column
        else None
    )
    best_micro = (
        max(
            (v for v in micro_by_model.values() if v is not None),
            default=None,
        )
        if micro_average_column
        else None
    )

    extra_cols = (
        ("c" if macro_average_column else "")
        + ("c" if micro_average_column else "")
        + ("c" if show_rate_column else "")
    )
    model_column = "l|" if group_separators else "l"
    tabular_spec = model_column + "c" * len(groups) + extra_cols
    header_cells = [
        f"\\textbf{{{format_header(g, abbreviate=abbreviate_headers)}}}"
        for g in groups
    ]
    if macro_average_column:
        header_cells.append("\\textbf{Macro Avg.}")
    if micro_average_column:
        header_cells.append("\\textbf{Micro Avg.}")
    if show_rate_column:
        header_cells.append("\\textbf{Parse Fail.}")

    lines = [
        "% Requires \\usepackage{booktabs}, \\usepackage[table]{xcolor},"
        " \\usepackage{nicematrix}",
        "\\begin{table*}[ht]",
        "\\centering",
        "\\definecolor{numpertdeltaimprove}{RGB}{0,100,0}",
        "\\definecolor{numpertdeltadecline}{RGB}{180,35,24}",
        "\\small",
        f"\\begin{{NiceTabular}}{{{tabular_spec}}}[first-col]",
        "\\toprule",
        " & \\textbf{Model} & " + " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    clusters: list[tuple[str, list[str]]] = []
    for model_label in ordered_models:
        grp = _model_group(
            short_model_label(model_label, results_root, dir_map)
        )
        if clusters and clusters[-1][0] == grp:
            clusters[-1][1].append(model_label)
        else:
            clusters.append((grp, [model_label]))

    for cluster_idx, (grp, members) in enumerate(clusters):
        if cluster_idx > 0:
            lines.append("\\midrule")
        group_label = ROW_GROUP_LABELS.get(grp, grp)
        for row_idx, model_label in enumerate(members):
            if row_idx == 0:
                first_col = (
                    f"\\Block{{{len(members)}-1}}" f"{{\\rotate {group_label}}}"
                )
            else:
                first_col = ""
            row_cells = [first_col, _row_label(model_label)]
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

                delta_super = ""
                if (
                    group_name != ORIGINAL_GROUP_NAME
                    and own_original is not None
                ):
                    delta_super = format_delta(
                        own_original.accuracy,
                        score.accuracy,
                        digits,
                    )

                parse_annotation = ""
                if show_parse_failures and score.parse_failures and score.total:
                    pct = score.parse_failures / score.total * 100
                    parse_annotation = f"~({pct:.{digits}f}\\%)"

                cell += delta_super + parse_annotation

                row_cells.append(cell)

            own_original_acc = (
                own_original.accuracy if own_original is not None else None
            )
            if macro_average_column:
                row_cells.append(
                    _format_average_cell(
                        macro_by_model.get(model_label),
                        best_macro,
                        own_original_acc,
                        digits,
                    )
                )
            if micro_average_column:
                row_cells.append(
                    _format_average_cell(
                        micro_by_model.get(model_label),
                        best_micro,
                        own_original_acc,
                        digits,
                    )
                )
            if show_rate_column:
                rate = rate_by_model.get(model_label)
                row_cells.append(
                    f"{rate:.{digits}f}\\%" if rate is not None else "--"
                )

            lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{NiceTabular}",
        ]
    )

    caption_body = escape_latex(caption_text)
    if parse_failures_display == "caption":
        breakdown = [
            (
                f"{_row_label(model_label)}: "
                f"{rate_by_model[model_label]:.{digits}f}\\%"
            )
            for model_label in ordered_models
            if rate_by_model.get(model_label) is not None
        ]
        if breakdown:
            caption_body += (
                " Macro-averaged parse-failure rates: "
                + ", ".join(breakdown)
                + "."
            )
    lines.append(f"\\caption{{{caption_body}}}")
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
            metrics_path,
            class_label=args.class_label,
            original_class_label=args.original_class_label,
        )
        for model_label, metrics_path in selected_models.items()
    }

    group_reference_label = (
        baseline_model
        if baseline_model is not None
        else next(iter(model_scores))
    )
    reference_scores = model_scores[group_reference_label]
    groups = derive_selected_groups(
        reference_scores,
        exact_groups=args.group,
        source_files=args.source_file,
        source_file_patterns=args.source_file_pattern,
        perturb_types=args.perturb_type,
        include_original=args.include_original,
    )
    if baseline_model is not None:
        ensure_baseline_groups(model_scores, baseline_model, groups)

    name_map = (
        load_model_name_map(args.model_name_map)
        if args.model_name_map
        else None
    )
    dir_map = (
        load_model_dir_map(args.model_dir_map) if args.model_dir_map else None
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
        dir_map=dir_map,
        abbreviate_headers=args.abbreviate_headers,
        parse_failures_display=args.parse_failures_display,
        macro_average_column=args.macro_average_column,
        micro_average_column=args.micro_average_column,
        group_separators=args.group_separators,
    )
    write_output(args.output, latex)
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
