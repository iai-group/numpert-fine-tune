"""Remove entries from JSON dataset and add claim IDs.

Cleans datasets by removing entries with a specified label and assigns
sequential claim identifiers to remaining entries.
"""

import json
import argparse
from typing import Tuple, List, Dict, Any
from itertools import count

# Global counter for claim IDs (per execution)
_claim_counter = count(start=1)


def generate_claim_id(dataset_split) -> str:
    """Generate a sequential claim ID.

    Args:
        dataset_split: The dataset split (e.g., 'train', 'val', 'test') to
            include in the claim ID for traceability.

    Returns:
        str: A sequential claim identifier (e.g., claim_000001).
    """
    return f"{dataset_split}_{next(_claim_counter):06d}"


def remove_label(
    input_json: str, label_to_remove: str, dataset_split: str
) -> Tuple[str, int, int]:
    """Remove entries with label and add claim IDs.

    Filters JSON data to exclude entries with a given label and assigns
    sequential claim identifiers to remaining entries.
    """
    data: List[Dict[str, Any]] = json.loads(input_json)
    total_entries: int = len(data)

    filtered_data: List[Dict[str, Any]] = []
    for entry in data:
        if entry.get("label") != label_to_remove:
            entry["claim_id"] = generate_claim_id(dataset_split)
            filtered_data.append(entry)

    remaining_entries: int = len(filtered_data)
    dropped_entries: int = total_entries - remaining_entries

    return (
        json.dumps(filtered_data, indent=4),
        dropped_entries,
        remaining_entries,
    )


def main() -> None:
    """Parse arguments and execute label removal."""
    parser = argparse.ArgumentParser(
        description="Remove specified label from JSON data."
    )
    parser.add_argument("input_json", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("label_to_remove", type=str)
    parser.add_argument(
        "dataset_split", type=str, choices=["train", "val", "test"]
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        input_json = f.read()

    filtered_json, dropped, remaining = remove_label(
        input_json, args.label_to_remove, args.dataset_split
    )

    with open(args.output_json, "w") as f:
        f.write(filtered_json)

    print(f"{dropped} elements were dropped. {remaining} elements remain.")
    print(f"Filtered JSON saved to {args.output_json}")
    print(f"Dataset split: {args.dataset_split}")


if __name__ == "__main__":
    main()
