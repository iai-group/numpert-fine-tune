"""Orchestrate spaCy-based number annotation over a dataset split.

For each record, locate numbers in the claim and the evidence document with
a spaCy locator, segment each text into sentences, enrich every span with
sentence-window context, then structurally validate the spans. Produces a
`{split}_spacy.json` file in the same record shape as the LLM track so the
two can be compared with `compare_models.py`.
"""

import argparse
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.number_annotation.core.context import enrich
from src.number_annotation.core.segmentation import segment
from src.number_annotation.annotators.spacy_locator import SpacyNumberLocator
from src.number_annotation.core.validator import validate_spans

INPUT = "data/raw_data/verdict_stripped/binary_rederived/{split}_binary.json"
OUTPUT = "data/spacy_number_annotated/{split}_spacy.json"
DEFAULT_MODEL = "en_core_web_trf"


def _resolve_io(input_path, output_path, split):
    """Resolve (input, output) from --input/--output or a --split.

    Exactly one of `input_path` / `split` must be given. When `--input` is
    used, `--output` is required; with `--split` both default to the binary
    convention.
    """
    if bool(input_path) == bool(split):
        raise ValueError("Provide exactly one of --input or --split.")
    if input_path:
        if not output_path:
            raise ValueError("--output is required with --input.")
        return input_path, output_path
    return (
        INPUT.format(split=split),
        output_path or OUTPUT.format(split=split),
    )


def _annotate_text(text: str, locator: Any) -> Dict[str, Any]:
    """Locate, segment, enrich, and validate one text."""
    text = text or ""
    spans = locator.locate(text)
    spans = enrich(spans, segment(text))
    report = validate_spans(text, spans)
    return {
        "spans": [s.to_dict() for s in spans],
        "valid": report.ok,
        "mismatches": report.mismatches,
        "overlaps": report.overlaps,
    }


def annotate_record(
    record: Dict[str, Any], locator: Any
) -> Dict[str, Any]:
    """Annotate a single record's claim and evidence (spaCy track)."""
    claim_res = _annotate_text(record.get("claim", ""), locator)
    doc_res = _annotate_text(record.get("doc", ""), locator)

    out = dict(record)
    out["claim_numbers"] = claim_res["spans"]
    out["evidence_numbers"] = doc_res["spans"]
    out["_annotation"] = {
        "claim_spans_valid": claim_res["valid"],
        "evidence_spans_valid": doc_res["valid"],
        # spaCy offsets are exact, so there are no location failures; the
        # zero-valued fields keep parity with the LLM-track record shape
        # consumed by compare_models.py.
        "claim_failures": 0,
        "evidence_failures": 0,
        "claim_failure_details": [],
        "evidence_failure_details": [],
    }
    return out


def annotate_split(
    input_path: str, output_path: str, locator: Any
) -> Dict[str, int]:
    """Annotate every record in a split file and save the result."""
    data: List[Dict[str, Any]] = load_data(input_path)
    # The full QuanTemp dataset lacks claim_id (make_binary adds it); assign a
    # stable, addressable id by position when one is missing.
    for i, record in enumerate(data):
        if not record.get("claim_id"):
            record["claim_id"] = f"test_{i:06d}"
    annotated = [annotate_record(r, locator) for r in data]
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
    """Annotate a split using the spaCy number locator."""
    parser = argparse.ArgumentParser(
        description="Annotate numeric mentions via spaCy NER.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Resolve input/output from the binary_rederived convention.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit input JSON (e.g. the full data/raw_data/"
        "test_claims_quantemp.json). Requires --output.",
    )
    parser.add_argument(
        "--output", default=None, help="Explicit output JSON path."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="spaCy model name (e.g. en_core_web_trf, en_core_web_lg).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run on GPU (needed for transformer models, e.g. trf).",
    )
    args = parser.parse_args()

    input_path, output_path = _resolve_io(args.input, args.output, args.split)
    locator = SpacyNumberLocator(model=args.model, use_gpu=args.gpu)
    stats = annotate_split(input_path, output_path, locator)
    print(
        f"Annotated '{input_path}' -> '{output_path}' "
        f"(spaCy, {args.model}): {stats}"
    )


if __name__ == "__main__":
    main()
