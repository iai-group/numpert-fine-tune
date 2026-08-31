"""Derive binary (True/False) splits from stripped QuanTemp data.

The legacy ``data/raw_data/binary_data`` (deleted 2026-07-03) was verified
to be exactly **QuanTemp minus the Conflicting-labeled records** (identical
claim multisets per split) — but its docs came from the old
``remove_referance.py``, which corrupted every decimal (``5.4`` → ``5. 4``),
dropped each doc's last sentence, and deleted any sentence containing
true/false/correct/misleading. This module re-derives the binary splits from
the *stripped* QuanTemp output instead: a pure label filter plus a
deterministic ``claim_id`` (``{datasplit}_{id:06d}``) for downstream
compatibility. Run it once the stripping rules are final, then re-annotate.

Usage (per split, after ``clean_splits``):

    python -m src.data_preprocessing.raw_dataset_preprocessing.make_binary \
      --input \
        data/raw_data/verdict_stripped/rulebased/test_claims_quantemp.json \
      --output data/raw_data/verdict_stripped/binary_rederived/test_binary.json
"""

import argparse
import json
from typing import Any, Dict, List


def make_binary_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter out Conflicting records and attach a deterministic claim_id.

    Records are copied, not mutated; all QuanTemp fields (incl. ``id`` and
    ``datasplit``) are preserved for provenance.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        if str(rec.get("label", "")).strip().lower() == "conflicting":
            continue
        if rec.get("id") is None or rec.get("datasplit") is None:
            raise ValueError(
                "record lacks id/datasplit — run the Step-0 id assignment "
                f"first (claim: {str(rec.get('claim'))[:60]!r})"
            )
        new = dict(rec)
        new["claim_id"] = f"{rec['datasplit']}_{rec['id']:06d}"
        out.append(new)
    return out


def main() -> None:
    """CLI: derive one binary split from a (stripped) QuanTemp split."""
    parser = argparse.ArgumentParser(
        description="Filter a QuanTemp split to True/False records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="QuanTemp split JSON (normally stripped).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        records = json.load(fh)
    out = make_binary_records(records)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"{args.input}: {len(records)} -> {len(out)} binary records "
          f"({len(records) - len(out)} Conflicting dropped) -> {args.output}")


if __name__ == "__main__":
    main()
