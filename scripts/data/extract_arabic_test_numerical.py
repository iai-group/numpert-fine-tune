"""
Extract numerical claims from the new Arabic test dataset.

Steps
-----
1. Load fact_check_all_ar_20250401_20260421.json (2413 records, all False)
2. Filter: keep claims containing non-year digits
3. Remove URL-only claims
4. Convert to QuanTemp schema with claim_id
5. Save to data/arabic/arabic_test_numerical.json for manual review

NOTE: No True labels exist in this dataset. Only False-side perturbations
      will be possible.

Usage
-----
    python scripts/data/extract_arabic_test_numerical.py
"""

import json
import re
from pathlib import Path

ARABIC_DIR = Path("data/arabic")

DIGIT_RE = re.compile(r"\b\d[\d,\.]*\b|[٠١٢٣٤٥٦٧٨٩]")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
URL_RE = re.compile(r"https?://")


def is_numerical(claim: str) -> bool:
    if URL_RE.search(claim):
        return False
    without_years = YEAR_RE.sub("", claim)
    return bool(DIGIT_RE.search(without_years))


def load_and_filter() -> list:
    src = ARABIC_DIR / "fact_check_all_ar_20250401_20260421.json"
    if not src.exists():
        raise SystemExit(f"{src} not found")

    records = json.loads(src.read_text())
    print(f"  Total records: {len(records)}")

    label_counts: dict = {}
    for r in records:
        l = r.get("label", "?")
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  Labels: {label_counts}")

    numerical = [r for r in records if is_numerical(r.get("claim", ""))]
    print(f"  After digit filter (non-year): {len(numerical)}")

    no_doc = sum(1 for r in numerical if not r.get("doc", "").strip())
    print(f"  Missing doc: {no_doc}")

    kept = [r for r in numerical if r.get("doc", "").strip()]
    print(f"  Final kept: {len(kept)}")
    return kept


def to_quanTemp_schema(records: list) -> list:
    out = []
    for i, r in enumerate(records):
        label = r.get("label", "False")
        # Treat Half True/False as False
        if label == "Half True/False":
            label = "False"
        out.append(
            {
                "crawled_date": r.get("crawled_date", r.get("published", "")),
                "country_of_origin": r.get("country_of_origin", "unknown"),
                "label": label,
                "url": r.get("url", ""),
                "lang": r.get("lang", "ar"),
                "claim": r["claim"],
                "doc": r["doc"],
                "taxonomy_label": "numerical",
                "label_original": r.get("label_original", label.lower()),
                "claim_id": f"arabic_test_{i:06d}",
                "speaker": r.get("speaker", ""),
                "reporting_source": r.get("fact_source", ""),
            }
        )
    return out


def main() -> None:
    print("Loading and filtering Arabic test data...")
    records = load_and_filter()

    print("\nConverting to QuanTemp schema...")
    records = to_quanTemp_schema(records)

    out_path = ARABIC_DIR / "arabic_test_numerical.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(records)} records -> {out_path}")
    print("\nNext step: manual review of arabic_test_numerical.json to build")
    print("REJECT_INDICES, then run the full perturbation pipeline.")


if __name__ == "__main__":
    main()
