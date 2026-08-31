"""First live smoke run: annotate a small length/class-varied sample.

Run this BEFORE the full seed to eyeball annotation quality and token
cost. `main()` performs real API calls (claim + doc per record), so it
spends tokens; `build_smoke_input` is pure sampling/IO and is unit-tested
offline.

Two backends are supported via ``--backend``:

* ``claude`` (default) – drives Claude Code headlessly via the CLI,
  using the user's subscription/OAuth auth.
* ``vertex`` – calls the Anthropic/Google API directly via gcloud
  application-default credentials.  Requires
  ``pip install 'anthropic[vertex]'`` and
  ``gcloud auth application-default login``.
"""

import argparse

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.number_annotation.annotators.annotator import (
    DEFAULT_ANNOTATOR_MODEL,
    DEFAULT_VERTEX_MODEL,
    DEFAULT_VERTEX_REGION,
    ClaudeCodeAnnotator,
    VertexAnnotator,
)
from src.number_annotation.pipeline.run_annotation import annotate_split
from src.number_annotation.smoke.sample_seed import smoke_sample

INPUT = "data/{split}_smoketest.json"
SMOKE_INPUT = "data/spacy_number_annotated/_smoke_input.json"
# Output is model-suffixed so runs with different models don't clobber
# each other (the input sample is identical across models for a given
# --n/--seed, enabling apples-to-apples comparison).
SMOKE_OUTPUT = "data/spacy_number_annotated/smoke_{model}_four_samples.json"


def build_smoke_input(
    input_path: str, output_path: str, n: int, seed: int
) -> int:
    """Sample `n` length/class-varied records and write them out."""
    data = load_data(input_path)
    sample = smoke_sample(data, n, seed)
    save_data(sample, output_path)
    return len(sample)


def main() -> None:
    """Sample and annotate a small smoke set."""
    parser = argparse.ArgumentParser(
        description="Annotate a small smoke sample (Claude Code or Vertex AI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=["claude", "vertex"],
        default="claude",
        help=(
            "'claude' uses the Claude Code CLI (subscription auth); "
            "'vertex' uses the Anthropic API on Vertex AI (gcloud ADC)."
        ),
    )
    # Claude Code options
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model alias for the Claude Code backend (e.g. opus), "
            "or Vertex model ID (e.g. gemini-2.5-flash). "
            "Defaults to the backend's own default."
        ),
    )
    # Vertex-only options
    parser.add_argument(
        "--project",
        default=None,
        metavar="PROJECT_ID",
        help="GCP project ID (vertex backend only; defaults to gcloud config).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_VERTEX_REGION,
        help="Vertex AI region (vertex backend only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Sampling temperature (vertex backend only). "
            "Omit to use the model default."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        dest="max_tokens",
        help="Maximum output tokens per call (vertex backend only).",
    )
    args = parser.parse_args()

    if args.backend == "vertex":
        model = args.model or DEFAULT_VERTEX_MODEL
        annotator = VertexAnnotator(
            model=model,
            project_id=args.project,
            region=args.region,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        model = args.model or DEFAULT_ANNOTATOR_MODEL
        annotator = ClaudeCodeAnnotator(model=model)

    # Vertex model IDs may contain "@"; use only the part before it as a
    # filename slug so smoke outputs stay filename-safe.
    model_tag = model.split("@")[0]
    out_path = SMOKE_OUTPUT.format(model=model_tag)

    count = build_smoke_input(
        INPUT.format(split=args.split), SMOKE_INPUT, args.n, args.seed
    )
    stats = annotate_split(SMOKE_INPUT, out_path, annotator)
    print(f"Smoke sample: {count} records ({model}) -> {out_path}")
    print(f"Stats: {stats}")
    print(
        "STOP. Review the smoke output before any full seed "
        "annotation."
    )


if __name__ == "__main__":
    main()
