"""Sanity check: multi-file loading, original deduplication, and batch data.

Loads multiple perturbation JSON files (which share the same ``claim_id``s),
runs them through ``expand_perturbations``, and writes the combined result to
JSONL so you can inspect exactly what the batch sampler / trainer would
receive.

Key things this script reports:
  - Duplicate ``original_normalized`` rows per ``claim_id``
  - Perturbation-type coverage per ``claim_id`` across all source files
  - Label distribution before and after deduplication
  - A deduplicated JSONL output file ready for downstream use

Usage::

    # From project root (with venv active):
    python -m tests.sanity_checks.check_multi_file_batch_data

    # Or with custom file configs on the CLI:
    python -m tests.sanity_checks.check_multi_file_batch_data \
        "data/perturbed_data/train_split/approximation_True.json:type1,type2" \
        "data/perturbed_data/train_split/range_True.json:type1" \
        --output combined_batch_data.jsonl

    # Write with duplicates kept (no deduplication):
    python -m tests.sanity_checks.check_multi_file_batch_data \
        "data/perturbed_data/train_split/approximation_True.json" \
        "data/perturbed_data/train_split/numeration_True.json" \
        --keep-duplicate-originals
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
    parse_file_config,
)
from src.llm_finetune.utils.data_loading.perturbation_logic import (
    expand_perturbations,
)
from src.llm_finetune.utils.data_loading.hf_dataset_instructions import (
    load_json_files,
)


SEPARATOR = "=" * 72
_ORIGINAL = PerturbationType.ORIGINAL_NORMALIZED.value

# ── Default configs (mirrors a typical multi-file training setup) ──────────
DEFAULT_CONFIGS: List[str] = [
    "data/perturbed_data/train_split/approximation_True.json:type1,type2",
    "data/perturbed_data/train_split/range_True.json:type1,type2",
]

DEFAULT_OUTPUT = "combined_batch_data.jsonl"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_and_expand_all(
    configs: List[FileLoadConfig],
) -> List[Dict]:
    """Load raw data from each config, expand perturbations, and annotate.

    Each expanded row gets a ``source_file`` field so downstream analysis
    can trace where it came from.

    Returns:
        A single flat list of expanded examples from all files.
    """
    all_expanded: List[Dict] = []

    for cfg in configs:
        path = Path(cfg.path)
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue

        raw = load_json_files(str(path))
        expanded = expand_perturbations(raw, allowed_types=cfg.perturb_types)

        for ex in expanded:
            ex["source_file"] = path.name

        all_expanded.extend(expanded)

    return all_expanded


def find_duplicate_originals(
    examples: List[Dict],
) -> Dict[str, List[int]]:
    """Return claim_ids that have more than one ``original_normalized`` row.

    Returns:
        Mapping from claim_id → list of indices into *examples* that are
        duplicate originals.
    """
    originals_by_claim: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        if ex["perturbation_type"] == _ORIGINAL:
            originals_by_claim[ex["claim_id"]].append(idx)

    return {
        cid: idxs for cid, idxs in originals_by_claim.items() if len(idxs) > 1
    }


def deduplicate_originals(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate ``original_normalized`` rows, keeping the first.

    All perturbed rows are always kept.

    Returns:
        A new list with at most one original per ``claim_id``.
    """
    seen_originals: set = set()
    deduped: List[Dict] = []

    for ex in examples:
        if ex["perturbation_type"] == _ORIGINAL:
            if ex["claim_id"] in seen_originals:
                continue  # skip duplicate
            seen_originals.add(ex["claim_id"])
        deduped.append(ex)

    return deduped


def write_jsonl(examples: List[Dict], output_path: Path) -> None:
    """Write examples to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_per_file(configs: List[FileLoadConfig]) -> None:
    """Print per-file expansion summary (similar to check_file_load_config)."""
    for cfg in configs:
        path = Path(cfg.path)
        print(SEPARATOR)
        print(f"FILE: {path}")
        print(SEPARATOR)

        if not path.exists():
            print("  File not found — skipping.\n")
            continue

        raw = load_json_files(str(path))
        if cfg.perturb_types is None:
            filter_label = "ALL"
        else:
            filter_label = ", ".join(t.value for t in cfg.perturb_types)

        expanded = expand_perturbations(raw, allowed_types=cfg.perturb_types)
        type_counts = Counter(ex["perturbation_type"] for ex in expanded)

        print(f"  Raw examples: {len(raw)}")
        print(f"  Filter: {filter_label}")
        print(f"  Expanded examples: {len(expanded)}")
        print()
        print("  Perturbation types:")
        for ptype, count in sorted(type_counts.items()):
            print(f"    {ptype:<30s}  {count:>5d}")
        print()


def report_combined(
    all_expanded: List[Dict],
    deduped: List[Dict],
    duplicates: Dict[str, List[int]],
) -> None:
    """Print combined summary with deduplication analysis."""
    print(SEPARATOR)
    print("  COMBINED ANALYSIS")
    print(SEPARATOR)
    print()

    # ── Overall counts ────────────────────────────────────────────────
    total_originals = sum(
        1 for ex in all_expanded if ex["perturbation_type"] == _ORIGINAL
    )
    total_perturbed = len(all_expanded) - total_originals
    unique_claims = len({ex["claim_id"] for ex in all_expanded})

    print(f"  Total expanded rows:       {len(all_expanded)}")
    print(f"  Unique claim_ids:          {unique_claims}")
    print(f"  Original rows (raw):       {total_originals}")
    print(f"  Perturbed rows:            {total_perturbed}")
    print()

    # ── Duplicate originals ───────────────────────────────────────────
    if duplicates:
        dup_count = sum(len(idxs) - 1 for idxs in duplicates.values())
        print(
            f"  Duplicate originals found: {len(duplicates)} claim_ids "
            f"({dup_count} extra rows)"
        )
        print()
        # Show first few
        show_limit = 10
        for i, (cid, idxs) in enumerate(sorted(duplicates.items())):
            if i >= show_limit:
                remaining = len(duplicates) - show_limit
                print(f"    ... and {remaining} more")
                break
            sources = [all_expanded[idx]["source_file"] for idx in idxs]
            print(f"    {cid}: {len(idxs)} originals from {sources}")
        print()
    else:
        print("  No duplicate originals found.")
        print()

    # ── After deduplication ───────────────────────────────────────────
    deduped_originals = sum(
        1 for ex in deduped if ex["perturbation_type"] == _ORIGINAL
    )
    deduped_perturbed = len(deduped) - deduped_originals

    print(f"  After deduplication:")
    print(f"    Total rows:    {len(deduped)}")
    print(f"    Originals:     {deduped_originals}")
    print(f"    Perturbed:     {deduped_perturbed}")
    print()

    # ── Per-type breakdown ────────────────────────────────────────────
    type_counts = Counter(ex["perturbation_type"] for ex in deduped)
    print("  Perturbation type distribution (deduplicated):")
    for ptype, count in sorted(type_counts.items()):
        print(f"    {ptype:<30s}  {count:>5d}")
    print()

    # ── Label distribution ────────────────────────────────────────────
    label_counts = Counter(ex["label"] for ex in deduped)
    print("  Label distribution (deduplicated):")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:<10s}  {count:>5d}")
    print()

    # ── Type coverage per claim ───────────────────────────────────────
    claim_types: Dict[str, set] = defaultdict(set)
    for ex in deduped:
        if ex["perturbation_type"] != _ORIGINAL:
            claim_types[ex["claim_id"]].add(ex["perturbation_type"])

    all_perturb_types = set()
    for types in claim_types.values():
        all_perturb_types |= types

    # Find claims missing types
    claims_missing_types: Dict[str, set] = {}
    for cid, types in claim_types.items():
        missing = all_perturb_types - types
        if missing:
            claims_missing_types[cid] = missing

    if claims_missing_types:
        print(
            f"  Claims with incomplete type coverage: "
            f"{len(claims_missing_types)}/{unique_claims}"
        )
        show_limit = 5
        for i, (cid, missing) in enumerate(
            sorted(claims_missing_types.items())
        ):
            if i >= show_limit:
                remaining = len(claims_missing_types) - show_limit
                print(f"    ... and {remaining} more")
                break
            print(f"    {cid}: missing {sorted(missing)}")
    else:
        print(
            f"  All {unique_claims} claims have full type coverage: "
            f"{sorted(all_perturb_types)}"
        )
    print()

    # ── Source file × type cross-tab ──────────────────────────────────
    print("  Source file × perturbation type (deduplicated):")
    file_type_counts = Counter(
        (ex.get("source_file", "unknown"), ex["perturbation_type"])
        for ex in deduped
    )
    current_file: Optional[str] = None
    for (sf, pt), count in sorted(file_type_counts.items()):
        if sf != current_file:
            print(f"    {sf}:")
            current_file = sf
        print(f"      {pt:<30s}  {count:>5d}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load multiple perturbation files, deduplicate originals, "
            "and write combined batch data."
        ),
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            "File config strings: path.json[:type1,type2,...].  "
            "Defaults to a built-in example pair if omitted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--keep-duplicate-originals",
        action="store_true",
        help="Write all rows including duplicate originals (no dedup).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:  # noqa: C901
    args = parse_args(argv)

    config_strings = args.configs if args.configs else DEFAULT_CONFIGS
    configs = [parse_file_config(s) for s in config_strings]
    output_path = Path(args.output)

    print()
    print(SEPARATOR)
    print("  Multi-File Batch Data Sanity Check")
    print(SEPARATOR)
    print()
    print(f"  Configs ({len(configs)}):")
    for i, cfg in enumerate(configs, 1):
        print(f"    [{i}] {cfg}")
    print(f"  Output: {output_path}")
    print(f"  Deduplicate originals: {not args.keep_duplicate_originals}")
    print()

    # ── Per-file reports ──────────────────────────────────────────────
    report_per_file(configs)

    # ── Combined analysis ─────────────────────────────────────────────
    all_expanded = load_and_expand_all(configs)

    if not all_expanded:
        print("  No data loaded — nothing to write.")
        sys.exit(1)

    duplicates = find_duplicate_originals(all_expanded)
    deduped = deduplicate_originals(all_expanded)

    report_combined(all_expanded, deduped, duplicates)

    # ── Write output ──────────────────────────────────────────────────
    output_data = all_expanded if args.keep_duplicate_originals else deduped
    write_jsonl(output_data, output_path)

    label = "with duplicates" if args.keep_duplicate_originals else "deduplicated"
    print(SEPARATOR)
    print(f"  Wrote {len(output_data)} rows ({label}) → {output_path}")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()
