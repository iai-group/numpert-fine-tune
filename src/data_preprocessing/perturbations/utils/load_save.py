"""Utility functions for loading and saving JSON data."""

from typing import Any, Dict, List
import json

# Keys to include in the output
INCLUDED_KEYS = {
    "crawled_date",
    "country_of_origin",
    "label",
    "url",
    "lang",
    "claim",
    "doc",
    "taxonomy_label",
    "label_original",
    "id",
    "normalized_claim",
    "claim_id",
}

# Alternatively, define the keys to exclude
EXCLUDED_KEYS = {"tokens", "ner_tags"}


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        List[Dict[str, Any]]: The loaded JSON data as a list of dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If the input is a dictionary (e.g., with URLs as keys),
        # convert it to a list
        if isinstance(data, dict):
            return list(data.values())
        return data


def save_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save JSON data to a specified file path.

    Args:
        data (List[Dict[str, Any]]): The data to be saved.
        file_path (str): The path to the output JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
