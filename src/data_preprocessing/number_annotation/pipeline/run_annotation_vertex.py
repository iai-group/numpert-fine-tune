"""Annotate a dataset split using Claude on Vertex AI (gcloud default auth).

Prerequisites
-------------
    pip install 'anthropic[vertex]'
    gcloud auth application-default login

Usage examples
--------------
    # Annotate the test split with defaults (claude-opus-4-5@20251101)
    python -m src.number_annotation.pipeline.run_annotation_vertex --split test

    # Sonnet, explicit project, limited to first 20 records for a smoke test
    python -m src.number_annotation.pipeline.run_annotation_vertex \\
        --split val \\
        --model claude-sonnet-4-5@20250929 \\
        --project my-gcp-project \\
        --limit 20

    # Haiku with temperature (only works on models that accept it)
    python -m src.number_annotation.pipeline.run_annotation_vertex \\
        --split train \\
        --model claude-haiku-3-5@20241022 \\
        --temperature 0.2 \\
        --max-tokens 2048
"""

import argparse

from src.number_annotation.annotators.annotator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_VERTEX_MODEL,
    DEFAULT_VERTEX_REGION,
    VertexAnnotator,
)
from src.number_annotation.pipeline.run_annotation import annotate_split

INPUT = "data/raw_data/verdict_stripped/binary_rederived/{split}_binary.json"
OUTPUT = "data/spacy_number_annotated/{split}_vertex_{model_tag}.json"


def _model_tag(model: str) -> str:
    """Shorten a Vertex model ID to a filename-safe slug."""
    # e.g. "claude-opus-4-5@20251101" → "claude-opus-4-5"
    # e.g. "gemini-2.5-flash" → "gemini-2.5-flash"
    return model.split("@")[0]


def main() -> None:
    """Annotate a split using Claude on Vertex AI."""
    parser = argparse.ArgumentParser(
        description="Annotate numeric mentions using Claude on Vertex AI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        help="Dataset split to annotate.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VERTEX_MODEL,
        help=(
            "Vertex AI Claude model ID, e.g. claude-opus-4-5@20251101 or "
            "claude-sonnet-4-5@20250929."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        metavar="PROJECT_ID",
        help=(
            "GCP project ID. Defaults to the project inferred from "
            "gcloud application-default credentials."
        ),
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_VERTEX_REGION,
        help="Vertex AI region.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Sampling temperature [0, 1]. Omit (default) to use the "
            "model default. Note: not accepted by Opus 4.7+ and later."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest="max_tokens",
        help="Maximum output tokens per API call.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Override the output file path. Default is "
            f"{OUTPUT.format(split='<split>', model_tag='<model>')}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N records (useful for smoke tests).",
    )
    args = parser.parse_args()

    annotator = VertexAnnotator(
        model=args.model,
        project_id=args.project,
        region=args.region,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    input_path = INPUT.format(split=args.split)
    output_path = args.output or OUTPUT.format(
        split=args.split,
        model_tag=_model_tag(args.model),
    )

    if args.limit is not None:
        # Wrap annotate_split to honour --limit without touching shared code.
        from typing import Any, Dict, List

        from src.data_preprocessing.perturbations_engine.utils.load_save import (
            load_data,
            save_data,
        )
        from src.number_annotation.pipeline.run_annotation import annotate_record

        data: List[Dict[str, Any]] = load_data(input_path)[: args.limit]
        annotated = [annotate_record(r, annotator) for r in data]
        save_data(annotated, output_path)
        stats = {
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
    else:
        stats = annotate_split(input_path, output_path, annotator)

    print(
        f"Annotated split '{args.split}' ({args.model}): {stats}\n"
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
