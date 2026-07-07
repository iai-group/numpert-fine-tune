"""
End-to-end perturbation pipeline for the Spanish numerical dataset.

Steps
-----
1. Load val_spanish_claims.json
2. Filter: keep True/False labels, remove Conflicting
3. Apply manual REJECT list (non-numerical claims identified by manual review)
4. Remove records without doc
5. Convert to QuanTemp schema with claim_id
6. Normalize number words to digits (reuses normalize_w2n)
7. Tag tokens with spaCy NER (en_core_web_sm)
8. Split into True / False files
9. Run all six perturbation types (reuses run_perturbations)
10. Write results to data/spanish/perturbed/val_split/

Usage
-----
    uv run --active python scripts/data/perturb_spanish_numerical.py
"""

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
)

warnings.filterwarnings("ignore")

SPANISH_DIR = Path("data/spanish")
TAGGED_DIR = SPANISH_DIR / "spacy_tagged"
PERTURBED_DIR = SPANISH_DIR / "perturbed"

# Indices within the label-filtered list (329 records after removing Conflicting)
# that were identified as non-numerical by manual review.
REJECT_INDICES = {
    0,
    1,
    4,
    5,
    9,
    10,
    11,
    12,
    17,
    19,
    22,
    23,
    24,
    29,
    33,
    39,
    42,
    43,
    51,
    54,
    55,
    56,
    58,
    62,
    69,
    71,
    81,
    86,
    87,
    92,
    93,
    95,
    97,
    101,
    102,
    110,
    111,
    113,
    115,
    124,
    130,
    134,
    139,
    142,
    146,
    147,
    152,
    154,
    156,
    159,
    161,
    162,
    164,
    167,
    168,
    171,
    177,
    178,
    179,
    187,
    190,
    198,
    199,
    207,
    211,
    213,
    214,
    215,
    216,
    225,
    226,
    231,
    238,
    239,
    240,
    246,
    248,
    250,
    260,
    265,
    266,
    268,
    273,
    274,
    285,
    288,
    290,
    291,
    292,
    294,
    306,
    308,
    315,
    318,
    323,
    326,
    327,
    328,
}


def load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit("Run: python -m spacy download en_core_web_sm")


def load_and_filter() -> list:
    src = SPANISH_DIR / "val_spanish_claims.json"
    if not src.exists():
        raise SystemExit(f"{src} not found")

    records = json.loads(src.read_text())
    # Keep only True/False labels
    label_filtered = [r for r in records if r.get("label") in ("True", "False")]
    print(f"  After label filter (drop Conflicting): {len(label_filtered)}")

    # Apply manual reject list and remove empty doc
    kept = []
    for i, r in enumerate(label_filtered):
        if i in REJECT_INDICES:
            continue
        if not r.get("doc", "").strip():
            continue
        kept.append(r)

    print(f"  After manual review + empty-doc removal: {len(kept)}")
    true_count = sum(1 for r in kept if r["label"] == "True")
    false_count = sum(1 for r in kept if r["label"] == "False")
    print(f"  True: {true_count}, False: {false_count}")
    return kept


def to_quanTemp_schema(records: list) -> list:
    out = []
    for i, r in enumerate(records):
        out.append(
            {
                "crawled_date": r.get("crawled_date", ""),
                "country_of_origin": r.get("country_of_origin", "unknown"),
                "label": r["label"],
                "url": r.get("url", ""),
                "lang": r.get("lang", "es"),
                "claim": r["claim"],
                "doc": r["doc"],
                "taxonomy_label": "numerical",
                "label_original": r.get("label", "").lower(),
                "claim_id": f"spanish_val_{i:06d}",
                "speaker": "",
                "reporting_source": "",
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

    print("\nLoading and filtering Spanish data...")
    records = load_and_filter()

    print("\nConverting to QuanTemp schema...")
    records = to_quanTemp_schema(records)

    # Save intermediate numerical file
    num_path = SPANISH_DIR / "spanish_numerical.json"
    num_path.parent.mkdir(parents=True, exist_ok=True)
    num_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  Saved {len(records)} records -> {num_path}")

    print("\nTagging entities...")
    true_records, false_records = tag_and_split(records, nlp)

    true_path = TAGGED_DIR / "val_true_tagged.json"
    false_path = TAGGED_DIR / "val_false_tagged.json"
    save_tagged(true_records, true_path)
    save_tagged(false_records, false_path)

    out_dir = PERTURBED_DIR / "val_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning perturbations -> {out_dir}")
    run_perturbations(str(true_path), str(false_path), str(out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
