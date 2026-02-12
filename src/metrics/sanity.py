#!/usr/bin/env python3
"""find_dups.py.

Usage examples:
  python find_dups.py input.jsonl
  python find_dups.py input.jsonl --no-normalize
  python find_dups.py input.jsonl --show-lines
  python find_dups.py input.jsonl --dedup-output dedup.jsonl
"""

import argparse
import json
import unicodedata
from collections import defaultdict

# common invisible/control chars to remove in normalization
_INVISIBLE = (
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u2060",  # WORD JOINER
    "\u2063",  # INVISIBLE SEPARATOR
    "\u00ad",  # SOFT HYPHEN
)


def normalize_id(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    for ch in _INVISIBLE:
        s = s.replace(ch, "")
    s = s.strip()
    # remove other Unicode C (control) category characters
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    return s


def bytes_hex(s: str) -> str:
    return " ".join(f"{b:02x}" for b in s.encode("utf-8", "surrogatepass"))


def debug_repr(s: str) -> str:
    return f"{repr(s)} | bytes: {bytes_hex(s)}"


def main():
    p = argparse.ArgumentParser(
        description="Find duplicate custom_id entries in a JSONL file."
    )
    p.add_argument("input", help="input.jsonl")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize custom_id before dedup/compare",
    )
    p.add_argument(
        "--show-lines",
        action="store_true",
        help="Also print the full JSON line for each duplicate occurrence",
    )
    p.add_argument(
        "--dedup-output",
        help="Write deduplicated output (first occurrence kept) to this file",
    )
    args = p.parse_args()

    occurrences = defaultdict(
        list
    )  # key -> list of (line_no, raw_id, full_line)
    total_lines = 0
    skipped_invalid = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            total_lines += 1
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue
            raw = obj.get("custom_id")
            if raw is None:
                continue
            key = str(raw) if args.no_normalize else normalize_id(raw)
            occurrences[key].append((lineno, str(raw), line.rstrip("\n")))

    unique_count = len(occurrences)
    print(
        f"Scanned {total_lines} lines, skipped {skipped_invalid} invalid JSON lines."
    )
    print(
        f"Unique custom_ids (by {'raw' if args.no_normalize else 'normalized'} key): {unique_count}"
    )
    # find duplicates
    duplicates = {k: v for k, v in occurrences.items() if len(v) > 1}
    if not duplicates:
        print("No duplicates found.")
        # optionally still write dedup if requested
    else:
        print(f"Found {len(duplicates)} duplicated id(s). Showing details:\n")
        # sort by most frequent first
        for key, rows in sorted(duplicates.items(), key=lambda kv: -len(kv[1])):
            print(f"---\nDuplicate key (used for grouping): {repr(key)}")
            print(f"Count: {len(rows)}")
            # show debug repr of the normalized key (so you can see bytes)
            print(f"Normalized/group key debug: {debug_repr(key)}")
            for lineno, raw, full_line in rows:
                print(f"  line {lineno}: raw id -> {debug_repr(raw)}")
                if args.show_lines:
                    snippet = (
                        full_line
                        if len(full_line) <= 400
                        else full_line[:400] + "..."
                    )
                    print(f"     JSON: {snippet}")
            print()

    # optional dedup write
    if args.dedup_output:
        # keep first occurrence for each key (preserve original full line text)
        chosen = []
        for key, rows in occurrences.items():
            if rows:
                chosen.append(rows[0][2])  # full_line
        with open(args.dedup_output, "w", encoding="utf-8") as out:
            for line in chosen:
                out.write(line.rstrip("\n") + "\n")
        print(f"Wrote {len(chosen)} unique objects to: {args.dedup_output}")


if __name__ == "__main__":
    main()
