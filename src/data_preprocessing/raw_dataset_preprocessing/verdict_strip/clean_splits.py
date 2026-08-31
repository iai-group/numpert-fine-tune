"""CLI: apply verdict stripping to split files, emit cleaned data + diff log."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    DROP_KEYS,
    DROP_REASONS,
    domain_of,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)


def clean_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (cleaned_records, diff_log). Records are copied, not mutated.

    Records keyed in rules.DROP_REASONS by (datasplit, id) are dropped
    wholesale BEFORE stripping: excluded from cleaned_records and logged as
    a {"dropped": True, "reason": ...} diff-log entry (no "removals" key --
    existing consumers read removals via .get and ignore these entries).
    """
    cleaned: List[Dict[str, Any]] = []
    difflog: List[Dict[str, Any]] = []
    for rec in records:
        out = dict(rec)
        domain = domain_of(rec.get("url", ""))
        # binary_data records key on claim_id instead of id; drops only
        # apply to int-`id` records (legacy claim_id strings like
        # "train_006419" never match drop keys).
        rec_id = rec.get("id", rec.get("claim_id"))
        drop_reason = DROP_REASONS.get((rec.get("datasplit"), rec_id))
        if drop_reason is not None:
            difflog.append(
                {
                    "id": rec_id,
                    "datasplit": rec.get("datasplit"),
                    "url": rec.get("url"),
                    "domain": domain,
                    "dropped": True,
                    "reason": drop_reason,
                }
            )
            continue
        doc = rec.get("doc") or ""
        res = strip(doc, domain)
        if res.removals:
            out["doc"] = res.cleaned
            difflog.append(
                {
                    "id": rec_id,
                    "datasplit": rec.get("datasplit"),
                    "url": rec.get("url"),
                    "domain": domain,
                    "rules": [r.rule for r in res.removals],
                    "removals": [
                        {
                            "rule": r.rule,
                            "mode": r.mode,
                            "category": r.category,
                            "span": list(r.span),
                            "removed_text": r.removed_text,
                        }
                        for r in res.removals
                    ],
                }
            )
        cleaned.append(out)
    return cleaned, difflog


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip verdicts from splits.")
    parser.add_argument(
        "files", nargs="+", help="Id-assigned split JSON files."
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw_data/verdict_stripped/rulebased",
        help="Directory for cleaned splits + diff_log.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the diff log only; skip writing cleaned splits.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    difflog_path = out_dir / "diff_log.jsonl"
    total_changed = 0
    total_dropped = 0
    matched_drops: set = set()
    splits_seen: set = set()

    with difflog_path.open("w", encoding="utf-8") as logf:
        for f in args.files:
            src = Path(f)
            records = load_data(str(src))
            splits_seen |= {r.get("datasplit") for r in records}
            cleaned, difflog = clean_records(records)
            dropped = [e for e in difflog if e.get("dropped")]
            matched_drops |= {(e["datasplit"], e["id"]) for e in dropped}
            n_changed = len(difflog) - len(dropped)
            total_changed += n_changed
            total_dropped += len(dropped)
            for line in difflog:
                logf.write(json.dumps(line, ensure_ascii=False) + "\n")
            if not args.dry_run:
                save_data(cleaned, str(out_dir / src.name))
            print(
                f"{src.name}: {n_changed} changed, {len(dropped)} dropped"
                f" / {len(records)} records"
            )
    # DROP_KEYS ids are positional (assign_ids.py): a key that matched no
    # record (pristine reorder, typo, subset run) is a silent no-op, so
    # warn about every unmatched key whose datasplit was processed.
    unmatched = {
        k for k in DROP_KEYS - matched_drops if k[0] in splits_seen
    }
    if unmatched:
        keys = ", ".join(f"{s}:{i}" for s, i in sorted(unmatched))
        print(
            f"WARNING: unmatched drop keys (no such record in the "
            f"processed splits; pristine order changed?): {keys}",
            file=sys.stderr,
        )
    print(
        f"total changed: {total_changed}, dropped: {total_dropped}; "
        f"diff log -> {difflog_path}"
    )


if __name__ == "__main__":
    main()
