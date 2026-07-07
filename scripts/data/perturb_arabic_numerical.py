"""
End-to-end perturbation pipeline for the Arabic numerical dataset.

Steps
-----
1. Load val_arabic_claim.json
2. Filter: keep only digit-containing claims (Western and Arabic-Indic digits)
3. Remove records without doc
4. Apply manual REJECT list (non-numerical claims identified by manual review)
5. Convert to QuanTemp schema with claim_id
6. Normalize number words to digits (reuses normalize_w2n)
7. Tag tokens with spaCy NER (en_core_web_sm detects digits as CARDINAL)
8. Split into True / False files
9. Run all six perturbation types (reuses run_perturbations)
10. Write results to data/arabic/perturbed/val_split/

Usage
-----
    uv run --active python scripts/data/perturb_arabic_numerical.py
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

ARABIC_DIR = Path("data/arabic")
TAGGED_DIR = ARABIC_DIR / "spacy_tagged"
PERTURBED_DIR = ARABIC_DIR / "perturbed"

DIGIT_RE = re.compile(r"\b\d[\d,\.]*\b|[٠١٢٣٤٥٦٧٨٩]")

# Indices within the digit-filtered list (508 records) that were identified as
# non-numerical by manual review. Reasons: year-only, date-only, age-as-identifier,
# competition-year label, channel/model identifier, birth date, phone number, etc.
REJECT_INDICES = {
    1,
    3,
    4,
    13,
    15,
    16,
    17,
    18,
    20,
    23,
    25,
    27,
    30,
    31,
    33,
    34,
    36,
    40,
    41,
    43,
    45,
    53,
    55,
    57,
    60,
    62,
    63,
    71,
    75,
    76,
    77,
    80,
    81,
    83,
    85,
    88,
    91,
    92,
    94,
    95,
    99,
    102,
    103,
    105,
    109,
    113,
    114,
    116,
    117,
    127,
    128,
    130,
    131,
    135,
    137,
    139,
    143,
    145,
    148,
    149,
    150,
    154,
    156,
    161,
    164,
    167,
    169,
    170,
    171,
    173,
    177,
    178,
    179,
    181,
    183,
    185,
    186,
    187,
    191,
    195,
    199,
    206,
    208,
    209,
    214,
    215,
    216,
    225,
    227,
    228,
    230,
    233,
    236,
    239,
    243,
    245,
    251,
    254,
    260,
    266,
    268,
    270,
    271,
    272,
    273,
    276,
    278,
    285,
    286,
    289,
    290,
    291,
    292,
    293,
    295,
    296,
    297,
    306,
    308,
    310,
    312,
    319,
    320,
    322,
    323,
    327,
    339,
    345,
    348,
    349,
    350,
    351,
    356,
    357,
    366,
    367,
    368,
    371,
    372,
    374,
    375,
    377,
    380,
    381,
    384,
    388,
    389,
    392,
    393,
    394,
    397,
    401,
    406,
    413,
    415,
    419,
    421,
    426,
    427,
    428,
    430,
    431,
    433,
    434,
    436,
    437,
    441,
    443,
    448,
    452,
    457,
    459,
    462,
    467,
    476,
    479,
    481,
    484,
    487,
    492,
    493,
    494,
    496,
    498,
    501,
    505,
    506,
    507,
}


def load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit("Run: python -m spacy download en_core_web_sm")


def load_and_filter() -> list:
    src = ARABIC_DIR / "val_arabic_claim.json"
    if not src.exists():
        raise SystemExit(f"{src} not found")

    records = json.loads(src.read_text())

    # Step 1: digit filter
    digit_filtered = [r for r in records if DIGIT_RE.search(r.get("claim", ""))]
    print(f"  After digit filter: {len(digit_filtered)}")

    # Step 2 & 3: apply reject list and remove empty doc
    kept = []
    for i, r in enumerate(digit_filtered):
        if i in REJECT_INDICES:
            continue
        if not r.get("doc", "").strip():
            continue
        kept.append(r)

    print(f"  After manual review + empty-doc removal: {len(kept)}")
    true_count = sum(1 for r in kept if r["label"] == "True")
    false_count = sum(1 for r in kept if r["label"] == "False")
    print(f"  True: {true_count}, False: {false_count}")
    if true_count < 10:
        print(
            f"  WARNING: only {true_count} True records — Arabic dataset lacks evidence "
            "text for most True claims; perturbation results will be skewed."
        )
    return kept


def to_quanTemp_schema(records: list) -> list:
    out = []
    for i, r in enumerate(records):
        out.append(
            {
                "crawled_date": r.get("crawled_date", r.get("date_stated", "")),
                "country_of_origin": r.get("country_of_origin", "unknown"),
                "label": r["label"],
                "url": r.get("url", ""),
                "lang": r.get("lang", "ar"),
                "claim": r["claim"],
                "doc": r["doc"],
                "taxonomy_label": "numerical",
                "label_original": r.get(
                    "label_original", r.get("label", "").lower()
                ),
                "claim_id": f"arabic_val_{i:06d}",
                "speaker": r.get("speaker", ""),
                "reporting_source": r.get("publisher_name", ""),
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

    print("\nLoading and filtering Arabic data...")
    records = load_and_filter()

    print("\nConverting to QuanTemp schema...")
    records = to_quanTemp_schema(records)

    # Save intermediate numerical file
    num_path = ARABIC_DIR / "arabic_numerical.json"
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
