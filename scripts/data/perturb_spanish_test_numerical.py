"""
End-to-end perturbation pipeline for the Spanish numerical test dataset.

Steps
-----
1. Load spanish_test_numerical.json (4927 pre-filtered numerical records)
2. Filter: keep True/False labels, drop Other/Conflicting
3. Remove records without doc
4. Convert to QuanTemp schema with claim_id
5. Normalize number words to digits (reuses normalize_w2n)
6. Tag tokens with spaCy NER (en_core_web_sm)
7. Split into True / False files
8. Run all six perturbation types (reuses run_perturbations)
9. Write results to data/spanish/perturbed/test_split/

Usage
-----
    PYTHONPATH=. .finetune/bin/python scripts/data/perturb_spanish_test_numerical.py
"""

import json
import re
import warnings
from pathlib import Path

import spacy

from src.data_preprocessing.perturbations.main_perturb_all import (
    run_perturbations,
)
from src.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy import (
    normalize_w2n,
    classify_entities,
)

warnings.filterwarnings("ignore")

SPANISH_DIR = Path("data/spanish")
TAGGED_DIR = SPANISH_DIR / "spacy_tagged_test"
PERTURBED_DIR = SPANISH_DIR / "perturbed"

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Numbers that are part of identifiers: adjacent to letters or connected by hyphen to letters
# Covers: 5G, COVID-19, H1N1, F-35, MP3, iPhone14, etc.
IDENTIFIER_NUM_RE = re.compile(
    r"[A-Za-záéíóúñüÁÉÍÓÚÑÜ]\d+"  # letter immediately before digit (H1N1, COVID19)
    r"|\d+[A-Za-záéíóúñüÁÉÍÓÚÑÜ]"  # digit immediately followed by letter (5G, MP3)
    r"|[A-Za-z]-\d+"  # letter-hyphen-digit (COVID-19, F-35)
)
# Standalone quantity: digit sequence not adjacent to letters (allows %, $, space, punctuation)
QUANTITY_RE = re.compile(
    r"(?<![A-Za-záéíóúñüÁÉÍÓÚÑÜ-])\d[\d,.]*(?![A-Za-záéíóúñüÁÉÍÓÚÑÜ])"
)


def has_meaningful_quantity(claim: str) -> bool:
    """Return True only if claim has at least one standalone numerical quantity.

    Filters out claims where every number is part of an identifier such as
    '5G', 'COVID-19', 'F-35', 'H1N1', etc.
    """
    text = YEAR_RE.sub("", claim)  # strip years first
    text = IDENTIFIER_NUM_RE.sub("", text)  # strip identifier-embedded numbers
    return bool(QUANTITY_RE.search(text))


def load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit("Run: python -m spacy download en_core_web_sm")


def load_and_filter() -> list:
    src = SPANISH_DIR / "spanish_test_numerical.json"
    if not src.exists():
        raise SystemExit(f"{src} not found")

    records = json.loads(src.read_text())
    print(f"  Total records: {len(records)}")

    kept = [
        r
        for r in records
        if r.get("label") in ("True", "False")
        and r.get("doc", "").strip()
        and has_meaningful_quantity(r.get("claim", ""))
    ]
    true_count = sum(1 for r in kept if r["label"] == "True")
    false_count = sum(1 for r in kept if r["label"] == "False")
    print(f"  After True/False + doc + meaningful-quantity filter: {len(kept)}")
    print(f"  True: {true_count}, False: {false_count}")
    return kept


def to_quanTemp_schema(records: list) -> list:
    out = []
    for i, r in enumerate(records):
        out.append(
            {
                "crawled_date": r.get("crawled_date", r.get("published", "")),
                "country_of_origin": r.get("country_of_origin", "unknown"),
                "label": r["label"],
                "url": r.get("url", ""),
                "lang": r.get("lang", "es"),
                "claim": r["claim"],
                "doc": r["doc"],
                "taxonomy_label": "numerical",
                "label_original": r.get("label_original", r["label"].lower()),
                "claim_id": f"spanish_test_{i:06d}",
                "speaker": r.get("speaker", ""),
                "reporting_source": r.get("fact_source", ""),
            }
        )
    return out


def tag_and_split(records: list, nlp) -> tuple[list, list]:
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


def main() -> None:
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = load_nlp()

    print("\nLoading and filtering Spanish test data...")
    records = load_and_filter()

    print("\nConverting to QuanTemp schema...")
    records = to_quanTemp_schema(records)

    print("\nTagging entities...")
    true_records, false_records = tag_and_split(records, nlp)

    true_path = TAGGED_DIR / "test_true_tagged.json"
    false_path = TAGGED_DIR / "test_false_tagged.json"
    save_tagged(true_records, true_path)
    save_tagged(false_records, false_path)

    out_dir = PERTURBED_DIR / "test_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning perturbations -> {out_dir}")
    run_perturbations(str(true_path), str(false_path), str(out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
