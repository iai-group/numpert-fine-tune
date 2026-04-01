"""Inspect sampled dataset JSON exported by the balancer sanity check.

This script reads the JSON written by
``tests.sanity_checks.check_dataset_balancer --output_json`` and reports:

  - overall dataset size and unique claim count
  - how many claim_ids were sampled multiple times
  - claim-level repetition details (original vs perturbed, type, method)
  - exact duplicated sampled rows, which is useful for verifying
    oversampling or repeated selection of the same perturbation

Usage::

    # Basic analysis
    python -m tests.sanity_checks.check_sampled_output_json \
        tests/sanity_checks/tmp.json

    # Show more repeated claims and duplicate rows
    python -m tests.sanity_checks.check_sampled_output_json \
        tests/sanity_checks/tmp.json --top 20

    # Inspect one specific claim id in detail
    python -m tests.sanity_checks.check_sampled_output_json \
        tests/sanity_checks/tmp.json --claim_id train_004019

    # Show claims with at least 10 sampled rows
    python -m tests.sanity_checks.check_sampled_output_json \
        tests/sanity_checks/tmp.json --min_multiplicity 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SEP = "=" * 80
THIN_SEP = "-" * 80


def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _subheader(title: str) -> None:
    print(f"\n  {THIN_SEP}")
    print(f"  {title}")
    print(f"  {THIN_SEP}")


def load_sampled_json(path: Path) -> list[dict]:
    """Load a sampled dataset JSON export."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    return data


def _claim_sort_key(item: tuple[str, list[dict]]) -> tuple[int, int, str]:
    claim_id, rows = item
    perturbed = sum(1 for row in rows if row.get("is_perturbed"))
    return (-len(rows), -perturbed, claim_id)


def report_overview(rows: list[dict]) -> dict[str, list[dict]]:
    """Print high-level repetition overview and return rows by claim."""
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        claim_id = row.get("source_claim_id", "(missing)")
        by_claim[claim_id].append(row)

    total_rows = len(rows)
    unique_claims = len(by_claim)
    repeated_claims = {
        claim_id: claim_rows
        for claim_id, claim_rows in by_claim.items()
        if len(claim_rows) > 1
    }
    originals = sum(1 for row in rows if not row.get("is_perturbed"))
    perturbed = total_rows - originals

    _header("SAMPLED OUTPUT OVERVIEW")
    print(f"  Input rows              : {total_rows}")
    print(f"  Unique claim_ids        : {unique_claims}")
    print(f"  Claims sampled >1 time  : {len(repeated_claims)}")
    print(f"  Originals               : {originals}")
    print(f"  Perturbed               : {perturbed}")
    max_rows = max(
        (len(claim_rows) for claim_rows in by_claim.values()),
        default=0,
    )
    print(f"  Max rows for one claim  : {max_rows}")

    multiplicity = Counter(len(claim_rows) for claim_rows in by_claim.values())
    _subheader("Claim sample multiplicity")
    for n_rows, count in sorted(multiplicity.items()):
        print(f"    {n_rows:>2d} row(s) per claim : {count:>6d}")

    return by_claim


def report_repeated_claims(
    by_claim: dict[str, list[dict]],
    top: int,
    claim_id_filter: str | None,
) -> None:
    """Print repeated-claim summaries."""
    repeated_items = [
        (claim_id, claim_rows)
        for claim_id, claim_rows in by_claim.items()
        if len(claim_rows) > 1
    ]

    if claim_id_filter is not None:
        repeated_items = [
            (claim_id, claim_rows)
            for claim_id, claim_rows in repeated_items
            if claim_id == claim_id_filter
        ]

    repeated_items.sort(key=_claim_sort_key)

    title = "Repeated claims"
    if claim_id_filter is not None:
        title = f"Repeated claim details: {claim_id_filter}"
    _header(title)

    if not repeated_items:
        print("  No repeated claims found.")
        return

    for claim_id, claim_rows in repeated_items[:top]:
        type_counter = Counter(
            row.get("type", "(missing)") for row in claim_rows
        )
        method_counter = Counter(
            row.get("perturbation_method") or "(original)" for row in claim_rows
        )
        file_counter = Counter(
            row.get("source_file", "(missing)") for row in claim_rows
        )

        print(f"\n  claim_id={claim_id}")
        print(f"    total rows     : {len(claim_rows)}")
        print(
            f"    originals      : "
            f"{sum(1 for row in claim_rows if not row.get('is_perturbed'))}"
        )
        print(
            f"    perturbed      : "
            f"{sum(1 for row in claim_rows if row.get('is_perturbed'))}"
        )
        print(f"    type counts    : {dict(type_counter)}")
        print(f"    method counts  : {dict(method_counter)}")
        print(f"    source files   : {dict(file_counter)}")


def report_high_multiplicity_claims(
    by_claim: dict[str, list[dict]],
    min_multiplicity: int,
    top: int,
) -> None:
    """Print claims whose sampled row count meets a threshold."""
    matching = [
        (claim_id, claim_rows)
        for claim_id, claim_rows in by_claim.items()
        if len(claim_rows) >= min_multiplicity
    ]
    matching.sort(key=_claim_sort_key)

    _header(f"Claims with at least {min_multiplicity} sampled rows")

    if not matching:
        print("  No claims matched this multiplicity threshold.")
        return

    for claim_id, claim_rows in matching[:top]:
        type_counter = Counter(
            row.get("type", "(missing)") for row in claim_rows
        )
        method_counter = Counter(
            row.get("perturbation_method") or "(original)" for row in claim_rows
        )

        print(f"\n  claim_id={claim_id}")
        print(f"    total rows     : {len(claim_rows)}")
        print(f"    type counts    : {dict(type_counter)}")
        print(f"    method counts  : {dict(method_counter)}")


def report_exact_duplicate_rows(rows: list[dict], top: int) -> None:
    """Print exact duplicated sampled rows.

    This captures actual oversampling of the same example, rather than just
    multiple different rows from the same claim.
    """
    signature_counter: Counter[
        tuple[
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
        ]
    ] = Counter()
    example_by_signature: dict[
        tuple[
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
        ],
        dict,
    ] = {}

    for row in rows:
        signature = (
            row.get("source_claim_id"),
            row.get("type"),
            row.get("perturbation_method"),
            row.get("label"),
            row.get("source_file"),
            row.get("text"),
        )
        signature_counter[signature] += 1
        example_by_signature.setdefault(signature, row)

    duplicates = [
        (signature, count)
        for signature, count in signature_counter.items()
        if count > 1
    ]
    duplicates.sort(key=lambda item: (-item[1], item[0][0] or ""))

    _header("Exact duplicated sampled rows")

    if not duplicates:
        print("  No exact duplicated sampled rows found.")
        return

    for signature, count in duplicates[:top]:
        row = example_by_signature[signature]
        print(f"\n  claim_id={row.get('source_claim_id')}  count={count}")
        print(f"    type              : {row.get('type')}")
        print(f"    method            : {row.get('perturbation_method')}")
        print(f"    label             : {row.get('label')}")
        print(f"    source_file       : {row.get('source_file')}")
        print(f"    text              : {row.get('text')}")


def main() -> None:
    """Analyze sampled output JSON for repeated claims and duplicates."""
    parser = argparse.ArgumentParser(
        description="Inspect sampled balanced dataset JSON output",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to JSON written by check_dataset_balancer --output_json",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many repeated claims / duplicate rows to show.",
    )
    parser.add_argument(
        "--claim_id",
        type=str,
        default=None,
        help="Optional specific claim_id to inspect in detail.",
    )
    parser.add_argument(
        "--min_multiplicity",
        type=int,
        default=10,
        help=(
            "Show a separate section listing claims with at least this many "
            "sampled rows."
        ),
    )
    args = parser.parse_args()

    rows = load_sampled_json(args.input_json)
    by_claim = report_overview(rows)
    report_repeated_claims(
        by_claim, top=args.top, claim_id_filter=args.claim_id
    )

    if args.claim_id is None:
        report_high_multiplicity_claims(
            by_claim,
            min_multiplicity=args.min_multiplicity,
            top=args.top,
        )

    if args.claim_id is None:
        report_exact_duplicate_rows(rows, top=args.top)

    print(f"\n{SEP}")
    print("  DONE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
