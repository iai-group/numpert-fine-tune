"""Apply negative number perturbations to percentage claims."""

import re
from typing import Dict, Any, List

from src.data_preprocessing.perturbations.utils.load_save import (
    INCLUDED_KEYS,
)

PERTUBATION_METHOD = "negative"


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
    for entity in sorted(entities, key=lambda x: x["start_char"], reverse=True):
        if entity["label"] == "PERCENT":
            start = entity["start_char"]
            end = entity["end_char"]
            original_text = perturbed_claim[start:end]
            number_match = re.search(r"-?\d+(\.\d+)?", original_text)
            if number_match and not number_match.group().startswith("-"):
                number_start, number_end = number_match.span()
                new_text = (
                    original_text[:number_start]
                    + "-"
                    + number_match.group()
                    + original_text[number_end:]
                )
                perturbed_claim = (
                    perturbed_claim[:start] + new_text + perturbed_claim[end:]
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
        perturbed_claim = negative_number_perturbation(claim, entities)

        if perturbed_claim != claim:
            label: str = entry.get("label", "UNKNOWN")
            new_entry = {
                key: entry[key] for key in INCLUDED_KEYS if key in entry
            }
            new_entry["pertubation_method"] = PERTUBATION_METHOD
            new_entry[f"perturb_type2_{label}"] = perturbed_claim
            output_data.append(new_entry)

    return output_data
