"""Orchestrate number annotation over a dataset split.

For each record, annotate both the claim and the evidence document, then
locate and validate the spans. Produces the reusable annotated corpus
(spec Deliverable 1) with `claim_numbers` and `evidence_numbers`.
"""

import argparse
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.number_annotation.annotators.annotator import (
    Annotator,
    ClaudeCodeAnnotator,
)
from src.number_annotation.core.locator import locate_numbers
from src.number_annotation.core.validator import validate_spans

INPUT = "data/raw_data/verdict_stripped/binary_rederived/{split}_binary.json"
OUTPUT = "data/spacy_number_annotated/{split}.json"


def _annotate_text(
    text: str, annotator: Annotator
) -> Dict[str, Any]:
    """Annotate one text and return spans + validity + failures."""
    raw = annotator.annotate(text or "")
    spans, failures = locate_numbers(text or "", raw)
    report = validate_spans(text or "", spans)
    return {
        "spans": [s.to_dict() for s in spans],
        "valid": report.ok,
        "n_failures": len(failures),
        "failure_details": [
            {
                "surface": f.surface,
                "context": f.context,
                "reason": f.reason,
                "duplicate_of": f.duplicate_of,
            }
            for f in failures
        ],
        "mismatches": report.mismatches,
        "overlaps": report.overlaps,
    }


def annotate_record(
    record: Dict[str, Any], annotator: Annotator
) -> Dict[str, Any]:
    """Annotate a single record's claim and evidence."""
    claim_res = _annotate_text(record.get("claim", ""), annotator)
    doc_res = _annotate_text(record.get("doc", ""), annotator)

    out = dict(record)
    out["claim_numbers"] = claim_res["spans"]
    out["evidence_numbers"] = doc_res["spans"]
    out["_annotation"] = {
        "claim_spans_valid": claim_res["valid"],
        "evidence_spans_valid": doc_res["valid"],
        "claim_failures": claim_res["n_failures"],
        "evidence_failures": doc_res["n_failures"],
        "claim_failure_details": claim_res["failure_details"],
        "evidence_failure_details": doc_res["failure_details"],
    }
    return out


def annotate_split(
    input_path: str, output_path: str, annotator: Annotator
) -> Dict[str, int]:
    """Annotate every record in a split file and save the result."""
    data: List[Dict[str, Any]] = load_data(input_path)
    annotated = [annotate_record(r, annotator) for r in data]
    save_data(annotated, output_path)

    return {
        "records": len(annotated),
        "claim_spans_valid": sum(
            1
            for r in annotated
            if r["_annotation"]["claim_spans_valid"]
        ),
        "evidence_spans_valid": sum(
            1
            for r in annotated
            if r["_annotation"]["evidence_spans_valid"]
        ),
    }


def main() -> None:
    """Annotate a split using the Claude Code annotator."""
    parser = argparse.ArgumentParser(
        description="Annotate numeric mentions in a dataset split."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Claude Code model alias (e.g. opus).",
    )
    args = parser.parse_args()

    annotator = (
        ClaudeCodeAnnotator(model=args.model)
        if args.model
        else ClaudeCodeAnnotator()
    )
    stats = annotate_split(
        INPUT.format(split=args.split),
        OUTPUT.format(split=args.split),
        annotator,
    )
    print(f"Annotated split '{args.split}': {stats}")


if __name__ == "__main__":
    main()
