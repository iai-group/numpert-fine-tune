"""Segment each record's evidence `doc` into indexed sentences.

Reuses the offset-exact pySBD `segment()` primitive. Each record gains a
`doc_sentences` list of {id, start, end, text}, where
`doc[start:end] == text`. The `claim` is left whole (it is the unit under
test). Output is the reusable sentence-segmented corpus for the
sentence-by-sentence number-tagging track.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.number_annotation.core.segmentation import segment

SPLIT_INPUT = "data/raw_data/verdict_stripped/binary_rederived/{split}_binary.json"


def segment_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `record` with a `doc_sentences` list added."""
    out = dict(record)
    seg = segment(record.get("doc", "") or "")
    out["doc_sentences"] = [
        {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
        for s in seg.sentences
    ]
    return out


def segment_dataset(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Segment every record in `input_path`, write to `output_path`."""
    data: List[Dict[str, Any]] = load_data(input_path)
    if limit is not None:
        data = data[:limit]

    segmented = [segment_record(r) for r in data]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_data(segmented, output_path)

    sentences_total = sum(len(r["doc_sentences"]) for r in segmented)
    empty_docs = sum(1 for r in segmented if not r["doc_sentences"])
    return {
        "records": len(segmented),
        "sentences_total": sentences_total,
        "empty_docs": empty_docs,
    }


def _resolve_input(
    input_path: Optional[str], split: Optional[str]
) -> str:
    """Return the input path from --input or a resolved --split.

    Exactly one of `input_path` / `split` must be provided.
    """
    if bool(input_path) == bool(split):
        raise ValueError(
            "Provide exactly one of --input or --split."
        )
    if input_path:
        return input_path
    return SPLIT_INPUT.format(split=split)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the segmentation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Segment each record's evidence doc into indexed sentences."
        )
    )
    parser.add_argument(
        "--input", default=None, help="Explicit input JSON path."
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "val", "test"],
        help="Resolve input from data/raw_data/verdict_stripped/binary_rederived/.",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON path."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records.",
    )
    return parser


def main() -> None:
    """Segment a dataset from the command line."""
    args = build_parser().parse_args()
    input_path = _resolve_input(args.input, args.split)
    stats = segment_dataset(input_path, args.output, limit=args.limit)
    print(f"Segmented '{input_path}' -> '{args.output}': {stats}")


if __name__ == "__main__":
    main()
