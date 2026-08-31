"""Re-run the hybrid pass for SPECIFIC records and patch the results back.

Two subcommands, used around a targeted `hybrid_strip` run:

  extract  Pull records by `<datasplit>:<id>` keys out of the rule-cleaned
           splits into one subset JSON (input for `hybrid_strip`).
  merge    Apply a subset hybrid output back into the main hybrid dir:
           per key, replace the record's `doc` in its split and replace (or
           append) its entry in `llm_diff_log.jsonl`.

Motivation: the 2026-07-07 greedy run (job 466238) left 11 records with
isolated client timeouts; re-running the full corpus for 11 records is
wasteful, so they are extracted, re-run with a longer timeout, and merged.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)

Key = Tuple[str, Any]
SPLITS = ("train", "val", "test")


def parse_keys(specs: List[str]) -> Set[Key]:
    """Parse `datasplit:id` specs (ids are ints in the splits)."""
    keys: Set[Key] = set()
    for spec in specs:
        split, _, rid = spec.partition(":")
        if not rid:
            raise ValueError(f"bad key spec {spec!r}, want datasplit:id")
        keys.add((split, int(rid)))
    return keys


def extract_records(split_dir: str, keys: Set[Key]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in SPLITS:
        path = Path(split_dir) / f"{split}_claims_quantemp.json"
        if not path.exists():
            continue
        for rec in load_data(str(path)):
            if (rec.get("datasplit"), rec.get("id")) in keys:
                out.append(rec)
    return out


def merge_subset(
    subset_records: List[Dict[str, Any]],
    subset_difflog: List[Dict[str, Any]],
    hybrid_dir: str,
) -> Dict[str, int]:
    """Patch subset results into the main hybrid dir. Returns counts."""
    by_key = {
        (r.get("datasplit"), r.get("id")): r for r in subset_records
    }
    docs_patched = 0
    for split in SPLITS:
        path = Path(hybrid_dir) / f"{split}_claims_quantemp.json"
        if not path.exists():
            continue
        records = load_data(str(path))
        changed = False
        for rec in records:
            sub = by_key.get((rec.get("datasplit"), rec.get("id")))
            if sub is not None and sub.get("doc") != rec.get("doc"):
                rec["doc"] = sub.get("doc")
                docs_patched += 1
                changed = True
        if changed:
            save_data(records, str(path))
    # difflog: drop stale entries for the patched keys, append fresh ones
    log_path = Path(hybrid_dir) / "llm_diff_log.jsonl"
    sub_keys = {
        (e.get("datasplit"), e.get("id")) for e in subset_difflog
    } | set(by_key)
    kept_lines: List[str] = []
    dropped = 0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if (e.get("datasplit"), e.get("id")) in sub_keys:
                dropped += 1
                continue
            kept_lines.append(line.rstrip("\n"))
    for e in subset_difflog:
        kept_lines.append(json.dumps(e, ensure_ascii=False))
    log_path.write_text(
        "\n".join(kept_lines) + "\n", encoding="utf-8"
    )
    return {
        "docs_patched": docs_patched,
        "difflog_dropped": dropped,
        "difflog_appended": len(subset_difflog),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="pull records into a subset file")
    p_ext.add_argument(
        "--from-dir",
        default="data/raw_data/verdict_stripped/rulebased",
        help="Directory with the rule-cleaned splits.",
    )
    p_ext.add_argument(
        "--keys", nargs="+", required=True, help="datasplit:id specs."
    )
    p_ext.add_argument("--out", required=True, help="Subset JSON path.")

    p_mrg = sub.add_parser("merge", help="patch subset output into hybrid/")
    p_mrg.add_argument(
        "--subset-dir", required=True,
        help="hybrid_strip --out-dir of the subset run.",
    )
    p_mrg.add_argument(
        "--subset-name", required=True,
        help="Basename of the subset split file inside --subset-dir.",
    )
    p_mrg.add_argument(
        "--hybrid-dir",
        default="data/raw_data/verdict_stripped/hybrid",
    )
    args = parser.parse_args()

    if args.cmd == "extract":
        keys = parse_keys(args.keys)
        records = extract_records(args.from_dir, keys)
        missing = keys - {
            (r.get("datasplit"), r.get("id")) for r in records
        }
        if missing:
            raise SystemExit(f"keys not found: {sorted(missing)}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_data(records, args.out)
        print(f"extracted {len(records)} records -> {args.out}")
    else:
        subset_records = load_data(
            str(Path(args.subset_dir) / args.subset_name)
        )
        log = Path(args.subset_dir) / "llm_diff_log.jsonl"
        with log.open(encoding="utf-8") as f:
            subset_difflog = [
                json.loads(line) for line in f if line.strip()
            ]
        counts = merge_subset(
            subset_records, subset_difflog, args.hybrid_dir
        )
        print(json.dumps(counts))


if __name__ == "__main__":
    main()
