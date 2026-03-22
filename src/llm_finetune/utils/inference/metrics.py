"""Metrics calculation for fact-checking inference results.

Provides structured metrics reporting with breakdowns by perturbation type,
source file, and combined groupings.  Designed to be called after inference with
a list of per-example result dicts.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Valid labels for fact-checking (JSON format).
VALID_LABELS = ['{"label": true}', '{"label": false}']

# Human-readable display names for label values.
LABEL_DISPLAY_NAMES = {
    '{"label": true}': "true",
    '{"label": false}': "false",
}


# ── dataclasses ──────────────────────────────────────────────────────────


@dataclass
class ClassificationScores:
    """Container for a single group's classification metrics."""

    group_name: str
    description: str = ""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    total: int = 0
    correct: int = 0
    parse_failures: int = 0
    classification_report_str: str = ""
    classification_report_dict: Dict = field(default_factory=dict)

    @property
    def valid(self) -> int:
        """Number of examples that were successfully parsed."""
        return self.total - self.parse_failures

    def to_dict(self) -> Dict:
        """Serialize to a plain dict with per-class classification report."""
        d: Dict = {
            "group_name": self.group_name,
        }
        if self.description:
            d["description"] = self.description
        d.update(
            {
                "accuracy": round(self.accuracy, 4),
                "f1_macro": round(self.f1_macro, 4),
                "f1_weighted": round(self.f1_weighted, 4),
                "precision_macro": round(self.precision_macro, 4),
                "recall_macro": round(self.recall_macro, 4),
                "total": self.total,
                "valid": self.valid,
                "correct": self.correct,
                "parse_failures": self.parse_failures,
            }
        )
        if self.classification_report_dict:
            d["classification_report"] = self.classification_report_dict
        return d


@dataclass
class MetricsReport:
    """Full metrics report with all groupings."""

    overall: Optional[ClassificationScores] = None
    per_perturbation_type: Dict[str, ClassificationScores] = field(
        default_factory=dict
    )
    per_source_file: Dict[str, ClassificationScores] = field(
        default_factory=dict
    )
    per_file_and_type: Dict[str, ClassificationScores] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict:
        """Serialize the entire report to a nested dict."""
        result: Dict = {}
        if self.overall is not None:
            result["overall"] = self.overall.to_dict()
        if self.per_perturbation_type:
            result["per_perturbation_type"] = {
                k: v.to_dict()
                for k, v in sorted(self.per_perturbation_type.items())
            }
        if self.per_source_file:
            result["per_source_file"] = {
                k: v.to_dict() for k, v in sorted(self.per_source_file.items())
            }
        if self.per_file_and_type:
            result["per_file_and_type"] = {
                k: v.to_dict()
                for k, v in sorted(self.per_file_and_type.items())
            }
        return result


# ── core computation ─────────────────────────────────────────────────────


def _compute_scores(
    group_name: str,
    true_labels: Sequence[str],
    predictions: Sequence[Optional[str]],
) -> ClassificationScores:
    """Compute classification scores for one group of examples.

    ``predictions`` may contain ``None`` values for parse failures
    — they are excluded from metric computation but counted in
    ``parse_failures``.

    Args:
        group_name: Human-readable identifier for this group.
        true_labels: Ground-truth labels (JSON-format strings).
        predictions: Model predictions (JSON-format strings or ``None``).

    Returns:
        Populated :class:`ClassificationScores`.
    """
    total = len(true_labels)
    valid_true: List[str] = []
    valid_pred: List[str] = []
    parse_failures = 0

    for gt, pred in zip(true_labels, predictions):
        if pred is None:
            parse_failures += 1
        else:
            valid_true.append(gt)
            valid_pred.append(pred)

    scores = ClassificationScores(
        group_name=group_name,
        total=total,
        parse_failures=parse_failures,
    )

    if not valid_true:
        return scores

    scores.accuracy = accuracy_score(valid_true, valid_pred)
    scores.f1_macro = f1_score(
        valid_true, valid_pred, average="macro", zero_division=0
    )
    scores.f1_weighted = f1_score(
        valid_true, valid_pred, average="weighted", zero_division=0
    )
    scores.precision_macro = precision_score(
        valid_true, valid_pred, average="macro", zero_division=0
    )
    scores.recall_macro = recall_score(
        valid_true, valid_pred, average="macro", zero_division=0
    )
    scores.correct = sum(1 for g, p in zip(valid_true, valid_pred) if g == p)
    scores.classification_report_str = classification_report(
        valid_true, valid_pred, labels=VALID_LABELS, zero_division=0
    )

    # Structured per-class report with readable keys
    raw_report = classification_report(
        valid_true,
        valid_pred,
        labels=VALID_LABELS,
        zero_division=0,
        output_dict=True,
    )
    readable_report: Dict = {}
    for key, value in raw_report.items():
        display_key = LABEL_DISPLAY_NAMES.get(key, key)
        if isinstance(value, dict):
            readable_report[display_key] = {
                k: round(v, 4) for k, v in value.items()
            }
        else:
            readable_report[display_key] = round(value, 4)
    scores.classification_report_dict = readable_report

    return scores


# ── description helpers ──────────────────────────────────────────────────

# Perturbation type category descriptions.
_PERTURB_TYPE_DESCRIPTIONS = {
    "original_normalized": "Baseline (unperturbed)",
    "perturb_type1": "Structure-preserving (label unchanged)",
    "perturb_type2": "Value-modifying (label flipped)",
    "perturb_type3_random": "Obfuscation: random digits (always False)",
    "perturb_type3_masked": "Obfuscation: masked with # (always False)",
    "perturb_type3_removed": "Obfuscation: numbers removed (always False)",
}

# Regex to extract the original-label indicator from source filenames
# e.g. "range_True.json" → "True", "numeration_False.json" → "False"
_SOURCE_LABEL_RE = re.compile(r"_(True|False)\.json$", re.IGNORECASE)


def _label_transition(original_label: str, perturb_type: str) -> str:
    """Return a label-transition string like 'True → True'."""
    orig = original_label.capitalize()  # "True" / "False"
    if perturb_type == "original_normalized":
        return f"{orig} → {orig}"
    elif perturb_type == "perturb_type1":
        # Structure-preserving: label unchanged
        return f"{orig} → {orig}"
    elif perturb_type == "perturb_type2":
        # Value-modifying: label flipped
        flipped = "False" if orig == "True" else "True"
        return f"{orig} → {flipped}"
    elif perturb_type.startswith("perturb_type3_"):
        # Type3 obfuscation: always False
        return f"{orig} → False"
    else:
        # Unknown perturbation type: no transition assumed
        return ""


def _describe_perturbation_type(perturb_type: str) -> str:
    """Generate a description for a perturbation-type group."""
    return _PERTURB_TYPE_DESCRIPTIONS.get(perturb_type, "")


def _describe_file_and_type(combo_key: str) -> str:
    """Generate a description for a source-file + perturbation-type group.

    Example input:  ``"range_True.json | perturb_type1"``
    Example output: ``"True → True — Structure-preserving (label unchanged)"``
    """
    parts = [p.strip() for p in combo_key.split("|")]
    if len(parts) != 2:
        return ""
    source_file, perturb_type = parts

    match = _SOURCE_LABEL_RE.search(source_file)
    if not match:
        # Can't determine original label from filename
        return _PERTURB_TYPE_DESCRIPTIONS.get(perturb_type, "")

    original_label = match.group(1)  # "True" or "False"
    transition = _label_transition(original_label, perturb_type)
    category = _PERTURB_TYPE_DESCRIPTIONS.get(perturb_type, "")
    if transition and category:
        return f"{transition} — {category}"
    if transition:
        return transition
    if category:
        return category
    return ""


# ── grouping helpers ─────────────────────────────────────────────────────


def _group_by_key(
    results: List[Dict],
    key: str,
) -> Dict[str, List[Dict]]:
    """Split *results* into sub-lists keyed by *key*."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in results:
        groups[item.get(key, "unknown")].append(item)
    return dict(groups)


# ── public API ───────────────────────────────────────────────────────────


def compute_metrics(
    results: List[Dict],
    include_per_type: bool = True,
    include_per_file: bool = True,
    include_per_file_and_type: bool = True,
) -> MetricsReport:
    """Compute a full :class:`MetricsReport` from inference results.

    Each item in *results* must have at minimum::

        {
            "label": str,            # ground-truth label (JSON format)
            "prediction": str|None,  # model prediction or None
            "perturbation_type": str, # perturbation type value
        }

    Optionally include ``"source_file": str`` for per-file breakdowns.

    Args:
        results: List of per-example result dicts.
        include_per_type: Compute per-perturbation-type breakdown.
        include_per_file: Compute per-source-file breakdown.
        include_per_file_and_type: Compute per-file+type breakdown.

    Returns:
        Populated :class:`MetricsReport`.
    """
    report = MetricsReport()

    # Overall
    true_labels = [r["label"] for r in results]
    predictions = [r["prediction"] for r in results]
    report.overall = _compute_scores("overall", true_labels, predictions)

    # Per perturbation type
    if include_per_type:
        for p_type, group in sorted(
            _group_by_key(results, "perturbation_type").items()
        ):
            gt = [r["label"] for r in group]
            pr = [r["prediction"] for r in group]
            scores = _compute_scores(f"perturbation_type={p_type}", gt, pr)
            scores.description = _describe_perturbation_type(p_type)
            report.per_perturbation_type[p_type] = scores

    # Per source file
    has_source_file = any("source_file" in r for r in results)
    if include_per_file and has_source_file:
        for src_file, group in sorted(
            _group_by_key(results, "source_file").items()
        ):
            gt = [r["label"] for r in group]
            pr = [r["prediction"] for r in group]
            report.per_source_file[src_file] = _compute_scores(
                f"source_file={src_file}", gt, pr
            )

    # Per file + perturbation type combination
    if include_per_file_and_type and has_source_file:
        combo_groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in results:
            combo_key = (
                f"{r.get('source_file', 'unknown')}"
                f" | {r.get('perturbation_type', 'unknown')}"
            )
            combo_groups[combo_key].append(r)

        for combo_key, group in sorted(combo_groups.items()):
            gt = [r["label"] for r in group]
            pr = [r["prediction"] for r in group]
            scores = _compute_scores(combo_key, gt, pr)
            scores.description = _describe_file_and_type(combo_key)
            report.per_file_and_type[combo_key] = scores

    return report


# ── printing / formatting ────────────────────────────────────────────────


def _print_section_header(title: str, width: int = 80) -> None:
    """Print a divider with a centred title."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _print_scores_table(
    scores_dict: Dict[str, ClassificationScores],
    show_report: bool = False,
) -> None:
    """Print a summary table for a dict of ClassificationScores."""
    header = (
        f"  {'Group':<40s} {'Acc':>7s} {'F1-M':>7s}"
        f" {'F1-W':>7s} {'Prec':>7s} {'Rec':>7s}"
        f" {'N':>5s} {'Fail':>5s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key in sorted(scores_dict):
        s = scores_dict[key]
        print(
            f"  {key:<40s} {s.accuracy:>7.4f} {s.f1_macro:>7.4f}"
            f" {s.f1_weighted:>7.4f} {s.precision_macro:>7.4f}"
            f" {s.recall_macro:>7.4f} {s.valid:>5d} {s.parse_failures:>5d}"
        )

    if show_report:
        for key in sorted(scores_dict):
            s = scores_dict[key]
            if s.classification_report_str:
                print(f"\n  Classification Report — {key}:")
                for line in s.classification_report_str.splitlines():
                    print(f"    {line}")


def print_metrics_report(
    report: MetricsReport,
    show_classification_reports: bool = False,
) -> None:
    """Pretty-print a :class:`MetricsReport` to stdout.

    Args:
        report: The computed metrics report.
        show_classification_reports: If ``True``, include the full
            sklearn classification report for each group.
    """
    if report.overall is not None:
        s = report.overall
        _print_section_header(
            f"OVERALL RESULTS  ({s.valid}/{s.total} parseable responses)"
        )
        print(f"  Accuracy:       {s.accuracy:.4f}")
        print(f"  F1 Macro:       {s.f1_macro:.4f}")
        print(f"  F1 Weighted:    {s.f1_weighted:.4f}")
        print(f"  Precision (M):  {s.precision_macro:.4f}")
        print(f"  Recall (M):     {s.recall_macro:.4f}")
        print(f"  Parse failures: {s.parse_failures}/{s.total}")

        if show_classification_reports and s.classification_report_str:
            print("\n  Classification Report:")
            for line in s.classification_report_str.splitlines():
                print(f"    {line}")

    if report.per_perturbation_type:
        _print_section_header("METRICS BY PERTURBATION TYPE")
        _print_scores_table(
            report.per_perturbation_type,
            show_report=show_classification_reports,
        )

    if report.per_source_file:
        _print_section_header("METRICS BY SOURCE FILE")
        _print_scores_table(
            report.per_source_file,
            show_report=show_classification_reports,
        )

    if report.per_file_and_type:
        _print_section_header("METRICS BY SOURCE FILE + PERTURBATION TYPE")
        _print_scores_table(
            report.per_file_and_type,
            show_report=show_classification_reports,
        )

    print()


# ── I/O ──────────────────────────────────────────────────────────────────


def save_metrics_report(
    report: MetricsReport,
    output_dir: str,
    filename: str = "metrics.json",
) -> str:
    """Save a :class:`MetricsReport` as JSON.

    Args:
        report: The metrics report to save.
        output_dir: Directory in which to write the file.
        filename: Name of the output JSON file.

    Returns:
        Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


def save_parse_failures(
    results: List[Dict],
    raw_outputs: List[Dict],
    output_dir: str,
    filename: str = "parse_failures.json",
) -> Optional[str]:
    """Save details of parse failures for debugging.

    Args:
        results: Per-example result dicts (with ``prediction`` key).
        raw_outputs: Raw inference outputs (parallel to *results*).
        output_dir: Directory in which to write the file.
        filename: Name of the output JSON file.

    Returns:
        Full path of the saved file, or ``None`` if there were no
        failures.
    """
    failures = [
        raw_outputs[i] for i, r in enumerate(results) if r["prediction"] is None
    ]
    if not failures:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    return path


def save_predictions_csv(
    results: List[Dict],
    raw_outputs: List[Dict],
    output_dir: str,
    filename: str = "predictions.csv",
) -> str:
    """Save a CSV of predictions for downstream analysis.

    Args:
        results: Per-example result dicts.
        raw_outputs: Raw inference outputs (parallel to *results*).
        output_dir: Directory in which to write the file.
        filename: Name of the output CSV file.

    Returns:
        Full path of the saved file.
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "claim_id": [r["claim_id"] for r in results],
            "perturbation_type": [r["perturbation_type"] for r in results],
            "source_file": [r.get("source_file", "") for r in results],
            "label": [r["label"] for r in results],
            "prediction": [
                r["prediction"]
                if r["prediction"] is not None
                else "PARSE_FAILURE"
                for r in results
            ],
            "parsed_successfully": [
                r["prediction"] is not None for r in results
            ],
            "raw_response": [o["raw_response"] for o in raw_outputs],
        }
    )

    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def build_result_records(
    test_data,
    raw_outputs: List[Dict],
) -> List[Dict]:
    """Build per-example result dicts from inference outputs.

    Uses the ``predicted_label`` field from *raw_outputs* (populated
    during inference) as the prediction for metrics computation.

    Args:
        test_data: HuggingFace dataset (or list of dicts) with at least
            ``claim_id``, ``label``, ``perturbation_type``, and
            optionally ``source_file``.
        raw_outputs: Inference output dicts containing at least
            ``predicted_label`` (str or ``None``).

    Returns:
        List of result dicts ready for :func:`compute_metrics`.
    """
    results: List[Dict] = []
    for i, output in enumerate(raw_outputs):
        record = {
            "claim_id": test_data[i]["claim_id"],
            "label": test_data[i]["label"],
            "perturbation_type": test_data[i]["perturbation_type"],
            "prediction": output.get("predicted_label"),
        }
        # Include source_file when available
        if "source_file" in test_data[i]:
            record["source_file"] = test_data[i]["source_file"]
        elif "source_file" in (
            test_data.column_names if hasattr(test_data, "column_names") else []
        ):
            record["source_file"] = test_data[i]["source_file"]
        results.append(record)
    return results
