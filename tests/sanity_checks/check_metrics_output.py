"""Sanity check: metrics computation from dummy inference results.

Creates a small set of dummy inference outputs with a mix of:
- Correct / incorrect predictions
- Parse failures (unparseable model responses)
- Multiple perturbation types (original, type1, type2, type3_random)
- Multiple source files (range_True.json, range_False.json)

Runs the metrics pipeline and writes ``inference_details.json`` and
``metrics.json`` to a temp directory, then prints the full output so you
can visually verify grouping, descriptions, and per-class reports.

Usage::

    python -m tests.sanity_checks.check_metrics_output

Expected output summary
-----------------------

The dummy data below defines **16 examples** across two source files and
four perturbation types.  The label-transition rules are:

    original_normalized : label unchanged (baseline)
    perturb_type1       : label unchanged (structure-preserving)
    perturb_type2       : label flipped   (value-modifying)
    perturb_type3_random: always False    (obfuscation)

Source file "range_True.json" contains claims whose *original* label is True.
Source file "range_False.json" contains claims whose *original* label is False.

Dummy examples and expected outcomes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 #  | source_file      | perturb_type     | gold  | pred  | ok?
----|------------------|------------------|-------|-------|-----
  0 | range_True.json  | original_norm.   | true  | true  | YES
  1 | range_True.json  | original_norm.   | true  | false | NO
  2 | range_True.json  | type1            | true  | true  | YES
  3 | range_True.json  | type1            | true  | false | NO
  4 | range_True.json  | type2            | false | false | YES
  5 | range_True.json  | type2            | false | true  | NO
  6 | range_True.json  | type3_random     | false | false | YES
  7 | range_True.json  | type3_random     | false | FAIL  | ---
  8 | range_False.json | original_norm.   | false | false | YES
  9 | range_False.json | original_norm.   | false | false | YES
 10 | range_False.json | type1            | false | false | YES
 11 | range_False.json | type1            | false | true  | NO
 12 | range_False.json | type2            | true  | true  | YES
 13 | range_False.json | type2            | true  | false | NO
 14 | range_False.json | type3_random     | false | false | YES
 15 | range_False.json | type3_random     | false | FAIL  | ---

Expected per-group metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

overall:
    total=16, valid=14, parse_failures=2, correct=9, accuracy=9/14≈0.6429

per_perturbation_type:
    original_normalized:  total=4, correct=3, accuracy=3/4=0.75
    perturb_type1:        total=4, correct=2, accuracy=2/4=0.50
    perturb_type2:        total=4, correct=2, accuracy=2/4=0.50
    perturb_type3_random: total=4, valid=2, correct=2, accuracy=2/2=1.00,
                          parse_failures=2

per_source_file:
    range_True.json:  total=8, valid=7, correct=4, accuracy=4/7≈0.5714
    range_False.json: total=8, valid=7, correct=5, accuracy=5/7≈0.7143

per_file_and_type (selected descriptions):
    range_True.json  | perturb_type1       → "True → True — …"
    range_True.json  | perturb_type2       → "True → False — …"
    range_False.json | perturb_type1       → "False → False — …"
    range_False.json | perturb_type2       → "False → True — …"
    range_False.json | perturb_type3_random→ "False → False — Obfuscation…"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.llm_finetune.utils.inference.metrics import (
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
)

LABEL_TRUE = '{"label": true}'
LABEL_FALSE = '{"label": false}'


def _build_dummy_data() -> tuple[list[dict], list[dict]]:
    """Return (inference_details, result_records) for the dummy scenario."""
    # Each tuple: (source_file, perturb_type, gold_label, raw_response)
    # raw_response is the model's raw text output.
    examples = [
        # ── range_True.json (original label = True) ──
        # original_normalized: gold=true
        (
            "range_True.json",
            "original_normalized",
            LABEL_TRUE,
            '{"label": true}',
        ),  # 0: correct
        (
            "range_True.json",
            "original_normalized",
            LABEL_TRUE,
            '{"label": false}',
        ),  # 1: wrong
        # perturb_type1 (structure-preserving): gold=true (T→T)
        (
            "range_True.json",
            "perturb_type1",
            LABEL_TRUE,
            '{"label": true}',
        ),  # 2: correct
        (
            "range_True.json",
            "perturb_type1",
            LABEL_TRUE,
            '{"label": false}',
        ),  # 3: wrong
        # perturb_type2 (value-modifying): gold=false (T→F)
        (
            "range_True.json",
            "perturb_type2",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 4: correct
        (
            "range_True.json",
            "perturb_type2",
            LABEL_FALSE,
            '{"label": true}',
        ),  # 5: wrong
        # perturb_type3_random (obfuscation): gold=false (always False)
        (
            "range_True.json",
            "perturb_type3_random",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 6: correct
        (
            "range_True.json",
            "perturb_type3_random",
            LABEL_FALSE,
            "I don't know",
        ),  # 7: PARSE FAIL
        # ── range_False.json (original label = False) ──
        # original_normalized: gold=false
        (
            "range_False.json",
            "original_normalized",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 8: correct
        (
            "range_False.json",
            "original_normalized",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 9: correct
        # perturb_type1 (structure-preserving): gold=false (F→F)
        (
            "range_False.json",
            "perturb_type1",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 10: correct
        (
            "range_False.json",
            "perturb_type1",
            LABEL_FALSE,
            '{"label": true}',
        ),  # 11: wrong
        # perturb_type2 (value-modifying): gold=true (F→T)
        (
            "range_False.json",
            "perturb_type2",
            LABEL_TRUE,
            '{"label": true}',
        ),  # 12: correct
        (
            "range_False.json",
            "perturb_type2",
            LABEL_TRUE,
            '{"label": false}',
        ),  # 13: wrong
        # perturb_type3_random (obfuscation): gold=false (always False)
        (
            "range_False.json",
            "perturb_type3_random",
            LABEL_FALSE,
            '{"label": false}',
        ),  # 14: correct
        (
            "range_False.json",
            "perturb_type3_random",
            LABEL_FALSE,
            "maybe true?",
        ),  # 15: PARSE FAIL
    ]

    inference_details = []
    result_records = []

    for i, (src, ptype, gold, raw) in enumerate(examples):
        # Parse the raw response the same way the real pipeline does
        predicted_label = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "label" in parsed:
                if isinstance(parsed["label"], bool):
                    predicted_label = (
                        f'{{"label": {str(parsed["label"]).lower()}}}'
                    )
        except (json.JSONDecodeError, ValueError):
            pass

        inference_details.append(
            {
                "index": i,
                "claim_id": f"claim_{i:03d}",
                "perturbation_type": ptype,
                "label": gold,
                "full_prompt": f"[prompt for example {i}]",
                "raw_response": raw,
                "parsed_successfully": predicted_label is not None,
                "predicted_label": predicted_label,
                "source_file": src,
            }
        )

        result_records.append(
            {
                "claim_id": f"claim_{i:03d}",
                "label": gold,
                "perturbation_type": ptype,
                "prediction": predicted_label,
                "source_file": src,
            }
        )

    return inference_details, result_records


def main() -> None:
    """Run the sanity check."""
    inference_details, result_records = _build_dummy_data()

    # ── Write inference_details.json ──
    with tempfile.TemporaryDirectory() as tmpdir:
        details_path = Path(tmpdir) / "inference_details.json"
        with open(details_path, "w") as f:
            json.dump(inference_details, f, indent=2)

        print("=" * 80)
        print("  INFERENCE DETAILS (first 3 entries + last parse failure)")
        print("=" * 80)
        for entry in inference_details[:3]:
            print(json.dumps(entry, indent=2))
        # Show a parse failure example
        failures = [
            e for e in inference_details if not e["parsed_successfully"]
        ]
        if failures:
            print("\n  ... (parse failure example):")
            print(json.dumps(failures[0], indent=2))

        # ── Compute and print metrics ──
        report = compute_metrics(result_records)

        print("\n")
        print_metrics_report(report, show_classification_reports=True)

        # ── Write metrics.json ──
        metrics_path = save_metrics_report(report, tmpdir)

        print("\n" + "=" * 80)
        print("  FULL metrics.json CONTENT")
        print("=" * 80)
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        print(json.dumps(metrics_data, indent=2))

        # ── Spot-check assertions ──
        print("\n" + "=" * 80)
        print("  SPOT-CHECK ASSERTIONS")
        print("=" * 80)

        overall = metrics_data["overall"]
        assert overall["total"] == 16, f"total: {overall['total']}"
        assert overall["valid"] == 14, f"valid: {overall['valid']}"
        assert (
            overall["parse_failures"] == 2
        ), f"failures: {overall['parse_failures']}"
        assert overall["correct"] == 9, f"correct: {overall['correct']}"
        print(
            "  [PASS] overall: total=16, valid=14, parse_failures=2, correct=9"
        )

        pt = metrics_data["per_perturbation_type"]
        assert pt["original_normalized"]["correct"] == 3
        assert pt["perturb_type1"]["correct"] == 2
        assert pt["perturb_type2"]["correct"] == 2
        assert pt["perturb_type3_random"]["correct"] == 2
        assert pt["perturb_type3_random"]["parse_failures"] == 2
        print(
            "  [PASS] per_perturbation_type correct counts and parse failures"
        )

        # Check descriptions exist on perturbation types
        for ptype in pt:
            assert "description" in pt[ptype], f"missing description: {ptype}"
        print("  [PASS] all perturbation types have descriptions")

        # Check file+type descriptions contain label transitions
        ft = metrics_data["per_file_and_type"]
        t1_true = ft["range_True.json | perturb_type1"]
        assert "True \u2192 True" in t1_true["description"]
        t2_true = ft["range_True.json | perturb_type2"]
        assert "True \u2192 False" in t2_true["description"]
        t1_false = ft["range_False.json | perturb_type1"]
        assert "False \u2192 False" in t1_false["description"]
        t2_false = ft["range_False.json | perturb_type2"]
        assert "False \u2192 True" in t2_false["description"]
        t3_false = ft["range_False.json | perturb_type3_random"]
        assert "False \u2192 False" in t3_false["description"]
        print("  [PASS] file+type descriptions have correct label transitions")

        # Check classification_report has per-class detail
        cr = overall.get("classification_report", {})
        assert "true" in cr, "missing 'true' class in classification_report"
        assert "false" in cr, "missing 'false' class in classification_report"
        assert "precision" in cr["true"]
        assert "recall" in cr["false"]
        print(
            "  [PASS] classification_report has per-class precision/recall/f1"
        )

        # Check inference_details structure
        for entry in inference_details:
            assert "parsed_successfully" in entry
            assert "predicted_label" in entry
            if entry["parsed_successfully"]:
                assert entry["predicted_label"] is not None
            else:
                assert entry["predicted_label"] is None
        print(
            "  [PASS] inference_details entries have"
            " parsed_successfully + predicted_label"
        )

        print("\n  All spot-checks passed!\n")


if __name__ == "__main__":
    main()
