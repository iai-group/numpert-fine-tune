"""Step 0: assign stable per-split ids to raw QuanTemp records.

Raw records carry no id. This pass adds `id` (the record's 0-based position
within its split) and `datasplit`. Idempotent: a record that already has an
`id` keeps it (its `datasplit` is filled in only if missing). Record order is
never changed, so the positional id stays valid downstream.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)

SPLITS = ("train", "val", "test")


def assign_ids(
    records: List[Dict[str, Any]], datasplit: str
) -> List[Dict[str, Any]]:
    """Add `id` (position) and `datasplit` to each record, in place."""
    for i, rec in enumerate(records):
        if rec.get("id") is None:
            rec["id"] = i
        rec.setdefault("datasplit", datasplit)
    return records


def _split_of(path: Path) -> str:
    name = path.name.lower()
    for s in SPLITS:
        if name.startswith(s):
            return s
    raise ValueError(f"Cannot infer datasplit from filename: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign id/datasplit in place."
    )
    parser.add_argument(
        "files", nargs="+", help="Split JSON files to enrich."
    )
    args = parser.parse_args()
    for f in args.files:
        path = Path(f)
        records = load_data(str(path))
        assign_ids(records, _split_of(path))
        save_data(records, str(path))
        print(f"assigned ids: {path} ({len(records)} records)")


if __name__ == "__main__":
    main()
