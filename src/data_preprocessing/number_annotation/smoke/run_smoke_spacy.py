"""Smoke-run the spaCy annotation track on the shared smoke sample.

Reuses the smoke input persisted by `run_smoke` (so the sample matches the
LLM smoke files exactly) and writes `smoke_spacy_four_samples.json`, which
slots into `compare_models.py` as model `spacy`. If the persisted input is
absent, it is built from the smoke source split first.
"""

import argparse
import os

from src.number_annotation.pipeline.run_annotation_spacy import annotate_split
from src.number_annotation.smoke.run_smoke import (
    INPUT,
    SMOKE_INPUT,
    build_smoke_input,
)
from src.number_annotation.annotators.spacy_locator import SpacyNumberLocator

DEFAULT_MODEL = "en_core_web_trf"
SMOKE_OUTPUT = "data/spacy_number_annotated/smoke_spacy_four_samples.json"


def main() -> None:
    """Sample (if needed) and annotate the smoke set with spaCy."""
    parser = argparse.ArgumentParser(
        description="Smoke-run the spaCy annotation track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="spaCy model name (e.g. en_core_web_trf, en_core_web_lg).",
    )
    args = parser.parse_args()

    if not os.path.exists(SMOKE_INPUT):
        build_smoke_input(
            INPUT.format(split=args.split), SMOKE_INPUT, args.n, args.seed
        )

    locator = SpacyNumberLocator(model=args.model)
    stats = annotate_split(SMOKE_INPUT, SMOKE_OUTPUT, locator)
    print(f"spaCy smoke ({args.model}) -> {SMOKE_OUTPUT}")
    print(f"Stats: {stats}")
    print(
        "Compare with: python -m src.number_annotation.eval.compare_models "
        "--models opus spacy --ref opus --html compare_spacy.html"
    )


if __name__ == "__main__":
    main()
