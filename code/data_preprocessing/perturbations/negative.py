"""Apply negative number perturbations to percentage claims."""

import json
import re
from typing import Dict, Any, List


# Numerical entity labels to consider
NUMERICAL_LABELS = {"PERCENT"}


def load_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded JSON data as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: Dict[str, Any], file_path: str) -> None:
    """Save JSON data to a specified file path.

    Args:
        data (Dict[str, Any]): The data to be saved.
        file_path (str): The path to the output JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def negative_number_perturbation(
    claim: str, entities: List[Dict[str, Any]]
) -> str:
    """Make numerical values negative within percent entities in the claim.

    Identify percentage entities in the claim and make the numerical
    values within them negative. This directly affects the truth value
    of the claim.

    Args:
        claim (str): The original claim text.
        entities (List[Dict[str, Any]]): List of entities with their positions.

    Returns:
        str: The claim with percentage numerical values made negative.
    """
    perturbed_claim = claim
    # Process entities in reverse order to maintain character positions
    for entity in sorted(entities, key=lambda x: x["start_char"], reverse=True):
        if entity["label"] == "PERCENT":
            start = entity["start_char"]
            end = entity["end_char"]
            original_text = perturbed_claim[start:end]
            # Find the numerical value in the original_text
            # This handles integers and decimals
            number_match = re.search(r"-?\d+(\.\d+)?", original_text)
            if number_match:
                number_start, number_end = number_match.span()
                number_text = number_match.group()
                # Check if number is already negative
                if not number_text.startswith("-"):
                    negative_number_text = "-" + number_text
                    # Replace the number in the original_text
                    new_text = (
                        original_text[:number_start]
                        + negative_number_text
                        + original_text[number_end:]
                    )
                    # Update perturbed_claim
                    perturbed_claim = (
                        perturbed_claim[:start]
                        + new_text
                        + perturbed_claim[end:]
                    )
    return perturbed_claim


def process_json(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process the input JSON data to apply negative number perturbations.

    Only includes claims that have been altered.
    """
    output_data: List[Dict[str, Any]] = []
    for entry in input_data:
        claim: str = entry["normalized_claim"]
        entities: List[Dict[str, Any]] = entry.get("entities", [])

        # Apply negative number perturbation
        perturbed_claim = negative_number_perturbation(claim, entities)

        # Include only altered claims
        if perturbed_claim != claim:
            # Start by copying all keys from the original entry
            new_entry = dict(entry)
            # Add the perturbed claim
            new_entry["negative_number_false"] = perturbed_claim

            # Remove unwanted keys if present
            for key_to_remove in ["ner_tags", "entities", "tokens"]:
                if key_to_remove in new_entry:
                    del new_entry[key_to_remove]

            output_data.append(new_entry)

    return output_data
