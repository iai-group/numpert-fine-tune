"""Remove entries with a specific label from a JSON dataset and add IDs.

This script processes a JSON dataset by removing entries with a specified label
and adding unique IDs to the remaining entries. It also provides a summary of
the number of entries removed and retained.
"""

import json
import argparse
from typing import Tuple, List, Dict, Any


def remove_label(input_json: str, label_to_remove: str) -> Tuple[str, int, int]:
    """Remove entries with the specified label from the JSON data and add IDs.

    Args:
        input_json (str): The JSON data as a string.
        label_to_remove (str): The label to remove from the dataset.

    Returns:
        Tuple[str, int, int]: A tuple containing:
            - The filtered JSON data as a string.
            - The number of entries removed.
            - The number of entries remaining.
    """
    # Load the JSON data
    data: List[Dict[str, Any]] = json.loads(input_json)

    # Count total entries before filtering
    total_entries: int = len(data)

    # Filter out the entries with the specified label and add IDs
    filtered_data: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if entry.get("label") != label_to_remove:
            entry["id"] = idx  # Add ID field
            filtered_data.append(entry)

    # Count remaining entries after filtering
    remaining_entries: int = len(filtered_data)

    # Count the number of dropped entries
    dropped_entries: int = total_entries - remaining_entries

    # Return the filtered dataset and the counts
    return (
        json.dumps(filtered_data, indent=4),
        dropped_entries,
        remaining_entries,
    )


def main() -> None:
    """Main function to remove a specific label from a JSON dataset.

    This function parses command-line arguments to specify the input JSON file,
    the output JSON file, and the label to remove. It then processes the dataset
    and saves the filtered results.
    """
    # Set up argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Remove specified label from JSON data."
    )
    parser.add_argument(
        "input_json", type=str, help="Path to the input JSON file."
    )
    parser.add_argument(
        "output_json", type=str, help="Path to save the filtered JSON output."
    )
    parser.add_argument(
        "label_to_remove",
        type=str,
        help="The label to remove from the dataset.",
    )

    # Parse the arguments
    args: argparse.Namespace = parser.parse_args()

    # Read the input JSON file
    with open(args.input_json, "r") as f:
        input_json: str = f.read()

    # Remove the specified label and get counts
    filtered_json, dropped, remaining = remove_label(
        input_json, args.label_to_remove
    )

    # Write the filtered JSON data to the output file
    with open(args.output_json, "w") as f:
        f.write(filtered_json)

    # Print the result message
    print(f"{dropped} elements were dropped. {remaining} elements remain.")
    print(f"Filtered JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()
