"""Sanity check: normalized-claim deduplication across input files.

Demonstrates that when the same ``original_normalized`` claim appears
in multiple source files, the inference result is computed **once** and
reused for subsequent occurrences.  Perturbed entries with the same
``claim_id`` and ``perturbation_type`` from different files are always
evaluated independently (they carry different claim text).

The script simulates the deduplication logic from
``unsloth_inference.py`` with dummy data and prints a step-by-step
trace showing which entries are evaluated vs. reused.

Usage::

    python -m tests.sanity_checks.check_normalized_claim_dedup

Expected output summary
-----------------------

Input: 10 rows across 3 source files (range_True.json,
approximation_True.json, numeration_True.json).

 #  | claim_id | perturb_type         | source_file          | action
----|----------|----------------------|----------------------|--------
  0 | c001     | original_normalized  | range_True.json      | EVALUATE
  1 | c001     | perturb_type1        | range_True.json      | EVALUATE
  2 | c001     | original_normalized  | approx_True.json     | REUSE #0
  3 | c001     | perturb_type1        | approx_True.json     | EVALUATE
  4 | c001     | original_normalized  | numer_True.json      | REUSE #0
  5 | c001     | perturb_type2        | numer_True.json      | EVALUATE
  6 | c002     | original_normalized  | range_True.json      | EVALUATE
  7 | c002     | perturb_type1        | range_True.json      | EVALUATE
  8 | c002     | original_normalized  | approx_True.json     | REUSE #6
  9 | c002     | original_normalized  | numer_True.json      | REUSE #6

Unique evaluations: 6   Reused: 4

Metrics for ``original_normalized`` must report total=6 (all rows),
but only 2 distinct model calls were needed for it.
"""

from __future__ import annotations

import json
import tempfile


from src.llm_finetune.utils.inference.metrics import (
    build_result_records,
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
)

from datasets import Dataset

SEPARATOR = "=" * 80


def _build_dummy_dataset() -> Dataset:
    """Build a 10-row dataset with overlapping original_normalized claims."""
    examples = [
        # ── range_True.json ──
        {
            "claim_id": "c001",
            "perturbation_type": "original_normalized",
            "label": "True",
            "source_file": "range_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: X is 100\nEvidence: ..."},
            ],
        },
        {
            "claim_id": "c001",
            "perturbation_type": "perturb_type1",
            "label": "True",
            "source_file": "range_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {
                    "role": "user",
                    "content": "Claim: X is 95-105\nEvidence: ...",
                },
            ],
        },
        # ── approximation_True.json (same c001, different perturbed text) ──
        {
            "claim_id": "c001",
            "perturbation_type": "original_normalized",
            "label": "True",
            "source_file": "approximation_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: X is 100\nEvidence: ..."},
            ],
        },
        {
            "claim_id": "c001",
            "perturbation_type": "perturb_type1",
            "label": "True",
            "source_file": "approximation_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {
                    "role": "user",
                    "content": "Claim: X is about 100\nEvidence: ...",
                },
            ],
        },
        # ── numeration_True.json (same c001 again) ──
        {
            "claim_id": "c001",
            "perturbation_type": "original_normalized",
            "label": "True",
            "source_file": "numeration_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: X is 100\nEvidence: ..."},
            ],
        },
        {
            "claim_id": "c001",
            "perturbation_type": "perturb_type2",
            "label": "False",
            "source_file": "numeration_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {
                    "role": "user",
                    "content": "Claim: X is one hundred\nEvidence: ...",
                },
            ],
        },
        # ── Another claim c002 with duplicates ──
        {
            "claim_id": "c002",
            "perturbation_type": "original_normalized",
            "label": "False",
            "source_file": "range_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: Y is 50\nEvidence: ..."},
            ],
        },
        {
            "claim_id": "c002",
            "perturbation_type": "perturb_type1",
            "label": "False",
            "source_file": "range_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {
                    "role": "user",
                    "content": "Claim: Y is 45-55\nEvidence: ...",
                },
            ],
        },
        {
            "claim_id": "c002",
            "perturbation_type": "original_normalized",
            "label": "False",
            "source_file": "approximation_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: Y is 50\nEvidence: ..."},
            ],
        },
        {
            "claim_id": "c002",
            "perturbation_type": "original_normalized",
            "label": "False",
            "source_file": "numeration_True.json",
            "conversations": [
                {"role": "system", "content": "Fact-checker"},
                {"role": "user", "content": "Claim: Y is 50\nEvidence: ..."},
            ],
        },
    ]
    return Dataset.from_list(examples)


def _simulate_dedup(ds: Dataset) -> tuple[list[int], dict[int, int]]:
    """Reproduce the dedup logic from unsloth_inference.py."""
    seen_original_claims: dict[str, int] = {}
    unique_indices: list[int] = []
    duplicate_map: dict[int, int] = {}

    for i in range(len(ds)):
        if ds[i]["perturbation_type"] == "original_normalized":
            cid = ds[i]["claim_id"]
            if cid not in seen_original_claims:
                seen_original_claims[cid] = i
                unique_indices.append(i)
            else:
                duplicate_map[i] = seen_original_claims[cid]
        else:
            unique_indices.append(i)

    return unique_indices, duplicate_map


def _simulate_inference(
    ds: Dataset,
    unique_indices: list[int],
    duplicate_map: dict[int, int],
) -> list[dict]:
    """Simulate inference with dummy predictions.

    - original_normalized claims predict correctly
    - perturb_type1 predicts correctly
    - perturb_type2 predicts incorrectly (wrong label)
    """
    raw_outputs: list[dict] = [{} for _ in range(len(ds))]

    for i in unique_indices:
        gold = ds[i]["label"]
        ptype = ds[i]["perturbation_type"]

        # Simulate model predictions
        if ptype == "perturb_type2":
            # Intentionally wrong for type2
            pred = "False" if gold == "True" else "True"
        else:
            # Correct for original + type1
            pred = gold

        raw_outputs[i] = {
            "index": i,
            "claim_id": ds[i]["claim_id"],
            "perturbation_type": ptype,
            "label": gold,
            "full_prompt": f"[prompt {i}]",
            "raw_response": pred,
            "parsed_successfully": True,
            "predicted_label": pred,
            "source_file": ds[i]["source_file"],
        }

    # Fill duplicates (same logic as unsloth_inference.py)
    for dup_idx, orig_idx in duplicate_map.items():
        orig = raw_outputs[orig_idx]
        raw_outputs[dup_idx] = {
            "index": dup_idx,
            "claim_id": ds[dup_idx]["claim_id"],
            "perturbation_type": ds[dup_idx]["perturbation_type"],
            "label": ds[dup_idx]["label"],
            "full_prompt": orig["full_prompt"],
            "raw_response": orig["raw_response"],
            "parsed_successfully": orig["parsed_successfully"],
            "predicted_label": orig["predicted_label"],
            "reused_from_index": orig_idx,
            "source_file": ds[dup_idx]["source_file"],
        }

    return raw_outputs


def main() -> None:
    """Run the sanity check."""
    ds = _build_dummy_dataset()
    unique_indices, duplicate_map = _simulate_dedup(ds)
    raw_outputs = _simulate_inference(ds, unique_indices, duplicate_map)

    # ── 1. Print the dedup trace ──
    print(SEPARATOR)
    print("  DEDUPLICATION TRACE")
    print(SEPARATOR)
    print(
        f"  {'#':>3}  {'claim_id':<10} {'perturb_type':<22}"
        f" {'source_file':<26} {'action'}"
    )
    print(
        f"  {'---':>3}  {'----------':<10} {'----------------------':<22}"
        f" {'--------------------------':<26} {'------'}"
    )

    for i in range(len(ds)):
        cid = ds[i]["claim_id"]
        ptype = ds[i]["perturbation_type"]
        src = ds[i]["source_file"]

        if i in duplicate_map:
            action = f"REUSE from #{duplicate_map[i]}"
        else:
            action = "EVALUATE"

        print(f"  {i:>3}  {cid:<10} {ptype:<22} {src:<26} {action}")

    n_eval = len(unique_indices)
    n_reuse = len(duplicate_map)
    print(f"\n  Unique evaluations: {n_eval}   Reused: {n_reuse}")
    print(f"  Total rows: {len(ds)}")

    # ── 2. Verify reused entries share predictions ──
    print(f"\n{SEPARATOR}")
    print("  REUSE VERIFICATION")
    print(SEPARATOR)

    for dup_idx, orig_idx in sorted(duplicate_map.items()):
        dup = raw_outputs[dup_idx]
        orig = raw_outputs[orig_idx]
        pred_match = dup["predicted_label"] == orig["predicted_label"]
        resp_match = dup["raw_response"] == orig["raw_response"]
        src_differs = dup["source_file"] != orig["source_file"]
        status = (
            "PASS" if (pred_match and resp_match and src_differs) else "FAIL"
        )
        print(
            f"  [{status}] Row {dup_idx} (src={dup['source_file']}) "
            f"reuses row {orig_idx} (src={orig['source_file']}): "
            f"pred_match={pred_match}, resp_match={resp_match}, "
            f"src_differs={src_differs}"
        )

    # ── 3. Verify perturbed entries are NOT reused ──
    print(f"\n{SEPARATOR}")
    print("  PERTURBED ENTRIES: NOT DEDUPLICATED")
    print(SEPARATOR)

    perturbed = [
        raw_outputs[i]
        for i in range(len(ds))
        if ds[i]["perturbation_type"] != "original_normalized"
    ]
    reused_perturbed = [e for e in perturbed if "reused_from_index" in e]
    print(f"  Total perturbed entries: {len(perturbed)}")
    print(f"  Perturbed entries reused: {len(reused_perturbed)}")
    assert len(reused_perturbed) == 0, "Perturbed entries must never be reused!"
    print("  [PASS] No perturbed entry was reused")

    # Show that same perturb_type1 from different files are both present
    # c001 has perturb_type1 from range_True and approx_True (different text)
    c001_t1 = [
        e
        for e in perturbed
        if e["perturbation_type"] == "perturb_type1" and e["claim_id"] == "c001"
    ]
    c001_t1_sources = [e["source_file"] for e in c001_t1]
    print(f"\n  c001 perturb_type1 entries: {len(c001_t1)}")
    print(f"  c001 perturb_type1 sources: {c001_t1_sources}")
    assert (
        len(c001_t1) == 2
    ), "Both c001 perturb_type1 entries must be evaluated"
    assert len(set(c001_t1_sources)) == 2, "They must come from different files"
    print(
        "[PASS] c001 perturb_type1 from different files evaluated independently"
    )

    # ── 4. Compute and display metrics ──
    result_records = build_result_records(ds, raw_outputs)
    report = compute_metrics(result_records)

    print(f"\n{SEPARATOR}")
    print("  METRICS REPORT")
    print(SEPARATOR)
    print_metrics_report(report, show_classification_reports=False)

    # ── 5. Spot-check metrics ──
    print(SEPARATOR)
    print("  SPOT-CHECK ASSERTIONS")
    print(SEPARATOR)

    metrics = report.to_dict()
    overall = metrics["overall"]

    # All 10 rows appear in overall counts
    assert overall["total"] == 10, f"Expected total=10, got {overall['total']}"
    print(f"  [PASS] overall total = {overall['total']} (all rows counted)")

    # original_normalized: 6 rows total (2 evaluated + 4 reused copies)
    pt = metrics["per_perturbation_type"]
    orig = pt["original_normalized"]
    assert orig["total"] == 6, f"Expected orig total=6, got {orig['total']}"
    # All original_normalized predictions are correct
    assert (
        orig["correct"] == 6
    ), f"Expected orig correct=6, got {orig['correct']}"
    print(
        f"  [PASS] original_normalized: total={orig['total']}, "
        f"correct={orig['correct']} (reused results are correct too)"
    )

    # perturb_type1: 3 entries (c001 from 2 files + c002 from 1), all correct
    t1 = pt["perturb_type1"]
    assert t1["total"] == 3, f"Expected type1 total=3, got {t1['total']}"
    assert t1["correct"] == 3, f"Expected type1 correct=3, got {t1['correct']}"
    print(
        f"  [PASS] perturb_type1: total={t1['total']}, correct={t1['correct']}"
    )

    # perturb_type2: 1 entry, intentionally wrong
    t2 = pt["perturb_type2"]
    assert t2["total"] == 1, f"Expected type2 total=1, got {t2['total']}"
    assert t2["correct"] == 0, f"Expected type2 correct=0, got {t2['correct']}"
    print(
        f"  [PASS] perturb_type2: total={t2['total']}, correct={t2['correct']}"
    )

    # Per source file: each file counts its own original_normalized rows
    sf = metrics["per_source_file"]
    assert sf["range_True.json"]["total"] == 4
    assert sf["approximation_True.json"]["total"] == 3
    assert sf["numeration_True.json"]["total"] == 3
    print(
        f"  [PASS] per_source_file totals: "
        f"range={sf['range_True.json']['total']}, "
        f"approx={sf['approximation_True.json']['total']}, "
        f"numer={sf['numeration_True.json']['total']}"
    )

    # Per file+type: original_normalized appears under each source file
    ft = metrics["per_file_and_type"]
    for src in [
        "range_True.json",
        "approximation_True.json",
        "numeration_True.json",
    ]:
        key = f"{src} | original_normalized"
        assert key in ft, f"Missing {key} in per_file_and_type"
    print("  [PASS] original_normalized reported under each source file")

    # Save full metrics JSON for inspection
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = save_metrics_report(report, tmpdir)
        with open(metrics_path) as f:
            full_metrics = json.load(f)
        print(f"\n{SEPARATOR}")
        print("  FULL metrics.json CONTENT")
        print(SEPARATOR)
        print(json.dumps(full_metrics, indent=2))

    print(f"\n{SEPARATOR}")
    print("  All spot-checks passed!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
