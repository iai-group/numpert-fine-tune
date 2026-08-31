"""Smoke-run the Gemini annotation track on four examples.

Reuses the shared smoke-input file produced by `run_smoke` so the sample is
identical across tracks (enabling apples-to-apples comparison).  If the
persisted input is absent it is built first.

Prerequisites::

    pip install google-genai
    gcloud auth application-default login

Then run::

    python -m src.number_annotation.smoke.run_smoke_gemini
"""

import argparse
import os

from src.number_annotation.annotators.annotator import (
    DEFAULT_GEMINI_MAX_TOKENS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_REGION,
    GeminiAnnotator,
)
from src.number_annotation.pipeline.run_annotation import annotate_split
from src.number_annotation.smoke.run_smoke import (
    INPUT,
    SMOKE_INPUT,
    build_smoke_input,
)

SMOKE_OUTPUT = "data/spacy_number_annotated/smoke_gemini_four_samples.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-run the Gemini annotation track (4 examples).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model ID on Vertex AI.",
    )
    parser.add_argument(
        "--project",
        default=None,
        metavar="PROJECT_ID",
        help="GCP project ID (defaults to gcloud config).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_GEMINI_REGION,
        help="Vertex AI region.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_GEMINI_MAX_TOKENS,
        dest="max_tokens",
    )
    args = parser.parse_args()

    if not os.path.exists(SMOKE_INPUT):
        count = build_smoke_input(
            INPUT.format(split=args.split), SMOKE_INPUT, args.n, args.seed
        )
        print(f"Built smoke input: {count} records -> {SMOKE_INPUT}")

    annotator = GeminiAnnotator(
        model=args.model,
        project_id=args.project,
        region=args.region,
        max_tokens=args.max_tokens,
    )

    model_tag = args.model.split("@")[0]
    out_path = SMOKE_OUTPUT.replace("gemini", model_tag) if "@" in args.model else SMOKE_OUTPUT

    stats = annotate_split(SMOKE_INPUT, out_path, annotator)
    print(f"Gemini smoke ({args.model}) -> {out_path}")
    print(f"Stats: {stats}")
    print(
        "Compare with: python -m src.number_annotation.eval.compare_models "
        "--models opus gemini --ref opus --html compare_gemini.html"
    )


if __name__ == "__main__":
    main()
