"""Sanity check: unperturbed-claim duplicates across a directory of JSON files.

This script scans all JSON files in a directory, reads each example,
and reports duplicate/consistency diagnostics for ``claim_id`` and
``normalized_claim``.

It is useful when multiple perturbation files share the same base claims and
we want to evaluate each normalized claim exactly once.

Usage::

    # Basic check
    python -m tests.sanity_checks.check_unperturbed_claims_dir \
        --input_dir data/perturbed_data/test_split

    # Recursive scan + write unique set
    python -m tests.sanity_checks.check_unperturbed_claims_dir \
        --input_dir data/perturbed_data/test_split \
        --recursive \
        --write_unique_jsonl results/unique_unperturbed_claims.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SEPARATOR = "=" * 80


@dataclass
class ClaimRecord:
    """One raw claim entry used for dedup/consistency reporting."""

    claim_id: str
    normalized_claim: str
    label: object
    source_file: str
    row_index: int


def _iter_json_files(input_dir: Path, recursive: bool) -> List[Path]:
    """Return sorted JSON files from directory."""
    if recursive:
        files = sorted(p for p in input_dir.rglob("*.json") if p.is_file())
    else:
        files = sorted(p for p in input_dir.glob("*.json") if p.is_file())
    return files


def _read_json_file(path: Path) -> List[dict]:
    """Read a JSON file containing either a list or a single object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return [data]


def _collect_claim_records(json_files: List[Path]) -> List[ClaimRecord]:
    """Load all files and extract (claim_id, normalized_claim) records."""
    rows: List[ClaimRecord] = []

    for path in json_files:
        data = _read_json_file(path)
        for i, ex in enumerate(data):
            claim_id = ex.get("claim_id")
            normalized_claim = ex.get("normalized_claim")
            if claim_id is None or normalized_claim is None:
                continue

            rows.append(
                ClaimRecord(
                    claim_id=str(claim_id),
                    normalized_claim=str(normalized_claim),
                    label=ex.get("label"),
                    source_file=path.name,
                    row_index=i,
                )
            )

    return rows


def _group_by_claim_id(
    rows: List[ClaimRecord],
) -> Dict[str, List[ClaimRecord]]:
    """Group claim records by claim_id."""
    by_id: Dict[str, List[ClaimRecord]] = defaultdict(list)
    for row in rows:
        by_id[row.claim_id].append(row)
    return by_id


def _build_unique_set(
    grouped: Dict[str, List[ClaimRecord]],
) -> List[ClaimRecord]:
    """Keep first occurrence per claim_id as unique set."""
    unique_rows: List[ClaimRecord] = []
    for claim_id in sorted(grouped):
        unique_rows.append(grouped[claim_id][0])
    return unique_rows


def _find_inconsistencies(
    grouped: Dict[str, List[ClaimRecord]],
) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Find claim_ids with conflicting normalized_claim and labels."""
    claim_text_conflicts: Dict[str, set] = {}
    label_conflicts: Dict[str, set] = {}

    for claim_id, records in grouped.items():
        claim_texts = {r.normalized_claim for r in records}
        labels = {str(r.label) for r in records}

        if len(claim_texts) > 1:
            claim_text_conflicts[claim_id] = claim_texts
        if len(labels) > 1:
            label_conflicts[claim_id] = labels

    return claim_text_conflicts, label_conflicts


def _print_summary(
    json_files: List[Path],
    rows: List[ClaimRecord],
    grouped: Dict[str, List[ClaimRecord]],
    unique_rows: List[ClaimRecord],
    claim_text_conflicts: Dict[str, set],
    label_conflicts: Dict[str, set],
) -> None:
    """Print duplicate and final-set diagnostics."""
    duplicates = {cid: recs for cid, recs in grouped.items() if len(recs) > 1}

    print()
    print(SEPARATOR)
    print("UNPERTURBED-CLAIM DIRECTORY SANITY CHECK")
    print(SEPARATOR)
    print(f"Input files scanned:      {len(json_files)}")
    print(f"Rows with required keys:  {len(rows)}")
    print(f"Unique claim_id count:    {len(grouped)}")
    print(f"Final unique set size:    {len(unique_rows)}")
    print()

    duplicate_rows = sum(len(v) - 1 for v in duplicates.values())
    print(f"Claim IDs with duplicates: {len(duplicates)}")
    print(f"Extra duplicate rows:      {duplicate_rows}")
    print()

    if duplicates:
        print("Top duplicate claim_ids (first 15):")
        top = sorted(duplicates.items(), key=lambda kv: len(kv[1]), reverse=True)
        for claim_id, recs in top[:15]:
            files = sorted({r.source_file for r in recs})
            print(
                f"  {claim_id}: {len(recs)} rows across {len(files)} files"
            )
        print()

    print(f"claim_id -> normalized_claim conflicts: {len(claim_text_conflicts)}")
    if claim_text_conflicts:
        for claim_id in sorted(claim_text_conflicts)[:10]:
            print(
                f"  {claim_id}: {len(claim_text_conflicts[claim_id])}"
                " normalized_claim variants"
            )
    print()

    print(f"claim_id -> label conflicts:            {len(label_conflicts)}")
    if label_conflicts:
        for claim_id in sorted(label_conflicts)[:10]:
            print(
                f"  {claim_id}: labels={sorted(label_conflicts[claim_id])}"
            )
    print()

    file_counts = Counter(r.source_file for r in rows)
    print("Rows per file (top 15):")
    for file_name, count in file_counts.most_common(15):
        print(f"  {file_name:<40s} {count:>6d}")
    print()


def _write_unique_jsonl(unique_rows: List[ClaimRecord], output_path: Path) -> None:
    """Write final unique set (one row per claim_id) to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in unique_rows:
            record = {
                "claim_id": row.claim_id,
                "normalized_claim": row.normalized_claim,
                "label": row.label,
                "source_file": row.source_file,
                "source_row_index": row.row_index,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory of perturbation JSON files and report "
            "duplicate/unique normalized-claim stats."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Scan JSON files recursively",
    )
    parser.add_argument(
        "--write_unique_jsonl",
        default=None,
        help=(
            "Optional path to write final unique set as JSONL "
            "(one row per claim_id)"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir must be an existing directory: {input_dir}")

    json_files = _iter_json_files(input_dir, recursive=args.recursive)
    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return

    rows = _collect_claim_records(json_files)
    grouped = _group_by_claim_id(rows)
    unique_rows = _build_unique_set(grouped)
    claim_text_conflicts, label_conflicts = _find_inconsistencies(grouped)

    _print_summary(
        json_files=json_files,
        rows=rows,
        grouped=grouped,
        unique_rows=unique_rows,
        claim_text_conflicts=claim_text_conflicts,
        label_conflicts=label_conflicts,
    )

    if args.write_unique_jsonl:
        output_path = Path(args.write_unique_jsonl)
        _write_unique_jsonl(unique_rows, output_path)
        print(f"Unique set written to: {output_path}")


if __name__ == "__main__":
    main()
