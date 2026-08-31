"""
Convert AVeriTeC train/dev splits to the QuanTemp raw-data schema,
keeping only numerical claims.

Numerical detection uses spaCy NER: a claim qualifies if it contains
at least one entity of type CARDINAL, PERCENT, MONEY, or QUANTITY,
OR if it contains a bare digit string not captured by NER.

Label mapping:
  Supported  -> True
  Refuted    -> False
  (Conflicting Evidence / Not Enough Evidence are discarded)

Evidence construction:
  Each question-answer pair is formatted as "Q: <question>\nA: <answer>"
  and joined with double newlines to form the doc field.

Usage:
  uv run --active python scripts/data/prepare_averitec_numerical.py
"""

import json
import re
import argparse
from pathlib import Path

import spacy


NUMERICAL_ENTS = {"CARDINAL", "PERCENT", "MONEY", "QUANTITY"}
DIGIT_RE = re.compile(r"\b\d[\d,\.]*\b")

LABEL_MAP = {
    "Supported": "True",
    "Refuted": "False",
    "Conflicting Evidence/Cherrypicking": "Conflicting",
}


def load_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


def is_numerical(claim: str, nlp) -> bool:
    """Return True if the claim contains numerical entities or bare digits."""
    doc = nlp(claim)
    if any(ent.label_ in NUMERICAL_ENTS for ent in doc.ents):
        return True
    return bool(DIGIT_RE.search(claim))


def build_doc(questions: list) -> str:
    """Combine Q&A pairs into a single evidence string."""
    parts = []
    for qa in questions:
        q = qa.get("question", "").strip()
        for ans in qa.get("answers", []):
            a = ans.get("answer", "").strip()
            if q and a:
                parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts)


def convert(records: list, split: str, nlp) -> list:
    out = []
    skipped_label = 0
    skipped_numerical = 0

    for i, rec in enumerate(records):
        raw_label = rec.get("label", "")
        if raw_label not in LABEL_MAP:
            skipped_label += 1
            continue

        claim = rec.get("claim", "").strip()
        if not is_numerical(claim, nlp):
            skipped_numerical += 1
            continue

        doc = build_doc(rec.get("questions", []))

        out.append(
            {
                "crawled_date": rec.get("claim_date", ""),
                "country_of_origin": (
                    rec.get("location_ISO_code") or ""
                ).lower()
                or "unknown",
                "label": LABEL_MAP[raw_label],
                "url": rec.get(
                    "fact_checking_article", rec.get("original_claim_url", "")
                ),
                "lang": "en",
                "claim": claim,
                "doc": doc,
                "taxonomy_label": "numerical",
                "label_original": raw_label.lower(),
                "claim_id": f"averitec_{split}_{i:06d}",
                "speaker": rec.get("speaker") or "",
                "reporting_source": rec.get("reporting_source") or "",
            }
        )

    print(
        f"  {split}: {len(records)} total | "
        f"{skipped_label} skipped (label) | "
        f"{skipped_numerical} skipped (not numerical) | "
        f"{len(out)} kept"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--averitec_dir",
        default="data/averitec",
        help="Directory containing train.json and dev.json",
    )
    parser.add_argument(
        "--out_dir",
        default="data/averitec",
        help="Directory to write output files",
    )
    args = parser.parse_args()

    averitec_dir = Path(args.averitec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy model...")
    nlp = load_nlp()

    for split in ("train", "dev"):
        src = averitec_dir / f"{split}.json"
        if not src.exists():
            print(f"  {src} not found, skipping")
            continue

        records = json.loads(src.read_text())
        converted = convert(records, split, nlp)

        dst = out_dir / f"averitec_{split}_numerical.json"
        dst.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
        print(f"  Written {len(converted)} records -> {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
