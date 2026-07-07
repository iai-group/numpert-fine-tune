"""
End-to-end perturbation pipeline for the AVeriTeC numerical dataset.

Steps
-----
1. Load averitec_{split}_numerical.json
2. Normalize number words to digits (reuses process_tag_claims_spacy logic)
3. Tag tokens with spaCy NER (en_core_web_sm)
4. Split into True / False files
5. Run all six perturbation types (reuses main_perturb_all.run_perturbations)
6. Write results to data/averitec/perturbed/{split}_split/

Usage
-----
    uv run --active python scripts/data/perturb_averitec_numerical.py
    uv run --active python scripts/data/perturb_averitec_numerical.py --splits dev
    uv run --active python scripts/data/perturb_averitec_numerical.py --splits train dev
"""

import argparse
import json
import warnings
from pathlib import Path

import spacy

from src.data_preprocessing.perturbations.main_perturb_all import (
    run_perturbations,
)
from src.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy import (
    normalize_w2n,
    classify_entities,
    get_target_entities,
)

warnings.filterwarnings("ignore")

AVERITEC_DIR = Path("data/averitec")
TAGGED_DIR = AVERITEC_DIR / "spacy_tagged"
PERTURBED_DIR = AVERITEC_DIR / "perturbed"


def load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit("Run: python -m spacy download en_core_web_sm")


def tag_and_split(records: list, nlp, split: str) -> tuple[list, list]:
    """Normalize numbers, run NER, return (true_records, false_records)."""
    print(f"  Normalizing {len(records)} claims...")
    records = normalize_w2n(records)

    print(f"  Tagging entities...")
    records = classify_entities(nlp, records)

    true_records = [r for r in records if r["label"] == "True"]
    false_records = [r for r in records if r["label"] == "False"]
    print(f"  True: {len(true_records)}, False: {len(false_records)}")
    return true_records, false_records


def save_tagged(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  Saved {len(records)} -> {path}")


def process_split(split: str, nlp) -> None:
    src = AVERITEC_DIR / f"averitec_{split}_numerical.json"
    if not src.exists():
        print(f"  {src} not found, skipping")
        return

    print(f"\n{'=' * 50}")
    print(f"Processing: {split}")
    print(f"{'=' * 50}")

    records = json.loads(src.read_text())

    true_records, false_records = tag_and_split(records, nlp, split)

    true_path = TAGGED_DIR / f"{split}_true_tagged.json"
    false_path = TAGGED_DIR / f"{split}_false_tagged.json"
    save_tagged(true_records, true_path)
    save_tagged(false_records, false_path)

    out_dir = PERTURBED_DIR / f"{split}_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning perturbations -> {out_dir}")
    run_perturbations(str(true_path), str(false_path), str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev"],
    )
    args = parser.parse_args()

    print("Loading spaCy model (en_core_web_sm)...")
    nlp = load_nlp()

    for split in args.splits:
        process_split(split, nlp)

    print("\nDone.")


if __name__ == "__main__":
    main()
