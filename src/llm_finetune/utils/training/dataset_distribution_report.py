"""Dataset distribution reporting for fine-tuning runs.

Computes a human-readable and machine-readable summary of the label,
perturbation type, source file, and perturbation method distributions
for a loaded HuggingFace dataset.  Mirrors the analysis performed by the
``check_dataset_balancer`` sanity script so that every training run produces
a permanent record of the data it was trained on.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from src.llm_finetune.utils.data_loading.perturbation_types import (
    PerturbationType,
)

ORIGINAL_NORMALIZED = PerturbationType.ORIGINAL_NORMALIZED.value
DISTRIBUTION_REPORT_FILENAME = "dataset_distribution.json"

SEP = "=" * 80
THIN_SEP = "-" * 60


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_label(raw: str) -> str:
    """Normalise a label value to ``"True"`` or ``"False"``.

    Handles both the plain string format (``"True"`` / ``"False"``) and the
    JSON format emitted by ``convert_to_conversations``
    (``'{"label": true}'`` / ``'{"label": false}'``).

    Args:
        raw: Raw label string from the dataset.

    Returns:
        ``"True"`` or ``"False"``.
    """
    if isinstance(raw, str):
        lower = raw.lower()
        if '"label": true' in lower or lower == "true":
            return "True"
        if '"label": false' in lower or lower == "false":
            return "False"
    return str(raw)


def _counter_table_lines(
    counter: Counter,
    total: int,
    col: int = 40,
) -> list[str]:
    """Format a counter as aligned table lines.

    Args:
        counter: Items and their counts.
        total: Grand total used for percentage calculation.
        col: Width of the key column.

    Returns:
        List of formatted strings (one per entry).
    """
    lines = []
    for key, count in counter.most_common():
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"    {str(key):<{col}}: {count:>6d}  ({pct:5.1f}%)")
    return lines


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def compute_distribution(dataset: Any) -> Dict[str, Any]:
    """Compute distribution statistics for a dataset split.

    Works with any object that supports column access via subscript and
    exposes a ``column_names`` attribute (HuggingFace ``Dataset``).

    Args:
        dataset: A HuggingFace Dataset (or compatible) returned by
            :func:`~pipeline.load_split`.  Expected columns:
            ``perturbation_type``, ``label``, ``source_file``,
            ``perturbation_method``, ``sample_type``.

    Returns:
        Dictionary with the following keys:

        - ``total``: Total number of examples.
        - ``label_distribution``: Counts for ``"True"`` and ``"False"``.
        - ``perturbation_type_distribution``: Counts per type string.
        - ``source_file_distribution``: Counts per source JSON file.
        - ``method_distribution``: Counts per perturbation method name.
        - ``originals``: Number of unperturbed (original) examples.
        - ``perturbed``: Number of perturbed examples.
        - ``perturb_to_orig_ratio``: ``perturbed / originals`` or ``null``.
        - ``label_by_perturbation``: Cross-tab of label × original/perturbed.
        - ``type_distribution``: Counts per directional type
          (``"Original"``, ``"T→T"``, ``"T→F"``, ``"F→F"``).
        - ``perturbation_class_composition``: Among perturbed examples,
          counts of T→T vs. combined False-producing (T→F + F→F) types.
        - ``false_pert_composition``: Among perturbed False examples,
          T→F vs. F→F counts and T→F ratio.
        - ``method_x_type_crosstab``: Nested dict
          ``{method: {sample_type: count}}``.
    """
    n = len(dataset)
    if n == 0:
        return {
            "total": 0,
            "label_distribution": {},
            "perturbation_type_distribution": {},
            "source_file_distribution": {},
            "method_distribution": {},
            "originals": 0,
            "perturbed": 0,
            "perturb_to_orig_ratio": None,
            "label_by_perturbation": {
                "original_true": 0,
                "original_false": 0,
                "perturbed_true": 0,
                "perturbed_false": 0,
            },
            "type_distribution": {},
            "perturbation_class_composition": {
                "T2T": 0,
                "false_pert": 0,
                "T2F": 0,
                "F2F": 0,
            },
            "false_pert_composition": {
                "T2F": 0,
                "F2F": 0,
                "T2F_ratio": None,
            },
            "method_x_type_crosstab": {},
        }

    columns = (
        dataset.column_names
        if hasattr(dataset, "column_names")
        else list(dataset[0].keys())
        if n > 0
        else []
    )

    # -- Labels ----------------------------------------------------------
    raw_labels = dataset["label"] if "label" in columns else []
    labels = [_parse_label(lbl) for lbl in raw_labels]

    # -- Perturbation types ----------------------------------------------
    perturb_types: list[str] = (
        list(dataset["perturbation_type"])
        if "perturbation_type" in columns
        else []
    )

    # -- Sample types (directional: Original, T→T, T→F, F→F) ------------
    sample_types: list[str] = (
        list(dataset["sample_type"]) if "sample_type" in columns else []
    )

    # -- Source files and methods ----------------------------------------
    source_files: list[str] = (
        list(dataset["source_file"]) if "source_file" in columns else []
    )
    methods: list[str] = (
        list(dataset["perturbation_method"])
        if "perturbation_method" in columns
        else []
    )

    # -- Original vs perturbed -------------------------------------------
    is_original = [pt == ORIGINAL_NORMALIZED for pt in perturb_types]
    n_original = sum(is_original)
    n_perturbed = n - n_original

    # -- Label × original/perturbed cross-tab ----------------------------
    n_true_orig = sum(
        1 for lbl, orig in zip(labels, is_original) if lbl == "True" and orig
    )
    n_false_orig = sum(
        1 for lbl, orig in zip(labels, is_original) if lbl == "False" and orig
    )
    n_true_pert = sum(
        1
        for lbl, orig in zip(labels, is_original)
        if lbl == "True" and not orig
    )
    n_false_pert = sum(
        1
        for lbl, orig in zip(labels, is_original)
        if lbl == "False" and not orig
    )

    ratio: Optional[float] = (
        n_perturbed / n_original if n_original > 0 else None
    )

    # -- Type distribution (T→T, T→F, F→F, Original) --------------------
    type_distribution: Dict[str, int] = (
        dict(Counter(sample_types).most_common()) if sample_types else {}
    )

    # -- Perturbation class composition (perturbed examples only) --------
    n_t2t = sum(
        1
        for st, orig in zip(sample_types, is_original)
        if not orig and st == "T\u2192T"
    )
    n_t2f = sum(
        1
        for st, orig in zip(sample_types, is_original)
        if not orig and st == "T\u2192F"
    )
    n_f2f = sum(
        1
        for st, orig in zip(sample_types, is_original)
        if not orig and st == "F\u2192F"
    )
    n_false_pert_types = n_t2f + n_f2f
    perturbation_class_composition: Dict[str, Any] = {
        "T2T": n_t2t,
        "false_pert": n_false_pert_types,
        "T2F": n_t2f,
        "F2F": n_f2f,
    }

    # -- False perturbation composition ----------------------------------
    t2f_ratio: Optional[float] = (
        n_t2f / n_false_pert_types if n_false_pert_types > 0 else None
    )
    false_pert_composition: Dict[str, Any] = {
        "T2F": n_t2f,
        "F2F": n_f2f,
        "T2F_ratio": t2f_ratio,
    }

    # -- Method × type cross-tab -----------------------------------------
    method_x_type: Dict[str, Dict[str, int]] = {}
    if sample_types and methods:
        for method, stype in zip(methods, sample_types):
            key = method if method else "(original)"
            if key not in method_x_type:
                method_x_type[key] = {}
            method_x_type[key][stype] = method_x_type[key].get(stype, 0) + 1

    return {
        "total": n,
        "label_distribution": dict(Counter(labels).most_common()),
        "perturbation_type_distribution": dict(
            Counter(perturb_types).most_common()
        ),
        "source_file_distribution": dict(Counter(source_files).most_common()),
        "method_distribution": dict(Counter(methods).most_common()),
        "originals": n_original,
        "perturbed": n_perturbed,
        "perturb_to_orig_ratio": ratio,
        "label_by_perturbation": {
            "original_true": n_true_orig,
            "original_false": n_false_orig,
            "perturbed_true": n_true_pert,
            "perturbed_false": n_false_pert,
        },
        "type_distribution": type_distribution,
        "perturbation_class_composition": perturbation_class_composition,
        "false_pert_composition": false_pert_composition,
        "method_x_type_crosstab": method_x_type,
    }


def _fmt_label_distribution(stats: Dict[str, Any], n: int) -> list[str]:
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Label distribution",
        f"  {THIN_SEP}",
    ]
    lines.extend(
        _counter_table_lines(Counter(stats.get("label_distribution", {})), n)
    )
    return lines


def _fmt_type_distribution(stats: Dict[str, Any], n: int) -> list[str]:
    type_dist = stats.get("type_distribution", {})
    if not type_dist:
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Type distribution",
        f"  {THIN_SEP}",
    ]
    lines.extend(_counter_table_lines(Counter(type_dist), n))
    return lines


def _fmt_label_by_perturbation(stats: Dict[str, Any], n: int) -> list[str]:
    cross = stats.get("label_by_perturbation", {})
    if not cross:
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Label \u00d7 original/perturbed",
        f"  {THIN_SEP}",
    ]
    pairs = [
        ("original_true", "Original  / True"),
        ("original_false", "Original  / False"),
        ("perturbed_true", "Perturbed / True"),
        ("perturbed_false", "Perturbed / False"),
    ]
    for key, label in pairs:
        count = cross.get(key, 0)
        pct = 100.0 * count / n if n else 0.0
        lines.append(f"    {label:<25s}: {count:>6d}  ({pct:5.1f}%)")
    return lines


def _fmt_perturbation_class_composition(
    stats: Dict[str, Any], n_pert: int
) -> list[str]:
    pcc = stats.get("perturbation_class_composition", {})
    if n_pert == 0 or not pcc:
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Perturbation class composition (perturbed only)",
        f"  {THIN_SEP}",
    ]
    for key, label in [
        ("T2T", "T\u2192T"),
        ("false_pert", "False pert (T\u2192F + F\u2192F)"),
    ]:
        count = pcc.get(key, 0)
        pct = 100.0 * count / n_pert if n_pert else 0.0
        lines.append(f"    {label:<35s}: {count:>6d}  ({pct:5.1f}%)")
    return lines


def _fmt_false_pert_composition(stats: Dict[str, Any]) -> list[str]:
    fpc = stats.get("false_pert_composition", {})
    n_total = fpc.get("T2F", 0) + fpc.get("F2F", 0)
    if n_total == 0:
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  False perturbation composition (T\u2192F vs F\u2192F)",
        f"  {THIN_SEP}",
    ]
    for key, label in [("T2F", "T\u2192F"), ("F2F", "F\u2192F")]:
        count = fpc.get(key, 0)
        pct = 100.0 * count / n_total
        lines.append(f"    {label:<10s}: {count:>6d}  ({pct:5.1f}%)")
    return lines


def _fmt_method_distribution(stats: Dict[str, Any], n: int) -> list[str]:
    method_dist = stats.get("method_distribution", {})
    if not any(m for m in method_dist if m):
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Perturbation method distribution",
        f"  {THIN_SEP}",
    ]
    lines.extend(_counter_table_lines(Counter(method_dist), n))
    return lines


def _fmt_source_file_distribution(stats: Dict[str, Any], n: int) -> list[str]:
    file_dist = stats.get("source_file_distribution", {})
    if not file_dist:
        return []
    lines = [
        "",
        f"  {THIN_SEP}",
        "  Source file distribution",
        f"  {THIN_SEP}",
    ]
    lines.extend(_counter_table_lines(Counter(file_dist), n))
    return lines


def _fmt_method_x_type_crosstab(stats: Dict[str, Any]) -> list[str]:
    crosstab = stats.get("method_x_type_crosstab", {})
    if not crosstab:
        return []
    all_types = sorted({st for counts in crosstab.values() for st in counts})
    col_w = 10
    hdr = f"  {'method':<22s}"
    for t in all_types:
        hdr += f"  {t:>{col_w}s}"
    hdr += f"  {'TOTAL':>{col_w}s}"

    lines = [
        "",
        f"  {THIN_SEP}",
        "  Method \u00d7 type cross-tab",
        f"  {THIN_SEP}",
        hdr,
        f"  {THIN_SEP}",
    ]
    for method in sorted(crosstab):
        row = f"  {method:<22s}"
        total = 0
        for t in all_types:
            c = crosstab[method].get(t, 0)
            total += c
            row += f"  {c:>{col_w}d}"
        row += f"  {total:>{col_w}d}"
        lines.append(row)
    return lines


def format_distribution_report(
    stats: Dict[str, Any],
    title: str = "DATASET DISTRIBUTION",
) -> str:
    """Return a human-readable distribution report string.

    Args:
        stats: Output of :func:`compute_distribution`.
        title: Header title for the report block.

    Returns:
        Formatted multi-line string ready for ``print()``.
    """
    n = stats.get("total", 0)
    n_pert = stats.get("perturbed", 0)
    ratio = stats.get("perturb_to_orig_ratio")
    ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"

    lines: list[str] = [
        "",
        SEP,
        f"  {title}",
        SEP,
        f"  Total examples : {n}",
        f"  Originals      : {stats.get('originals', 0)}",
        f"  Perturbed      : {n_pert}",
        f"  Pert/Orig ratio: {ratio_str}",
    ]
    lines += _fmt_label_distribution(stats, n)
    lines += _fmt_type_distribution(stats, n)
    lines += _fmt_label_by_perturbation(stats, n)
    lines += _fmt_perturbation_class_composition(stats, n_pert)
    lines += _fmt_false_pert_composition(stats)
    lines += _fmt_method_distribution(stats, n)
    lines += _fmt_source_file_distribution(stats, n)
    lines += _fmt_method_x_type_crosstab(stats)
    lines.append(SEP)
    return "\n".join(lines)


def save_distribution_report(
    train_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Save train and validation distribution statistics to a JSON file.

    Args:
        train_stats: Output of :func:`compute_distribution` for the
            training split.
        val_stats: Output of :func:`compute_distribution` for the
            validation split.
        output_path: Destination file path (created with parents).

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": train_stats, "val": val_stats}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return output_path
