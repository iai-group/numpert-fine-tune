"""Apply obfuscation perturbations to the claims in the dataset.

This module provides functions to perturb numerical claims in text by applying
random replacements, masking, and value removal. It processes claims and their
associated entities to generate perturbed datasets.
"""

import random
import string
from typing import Dict, Any, List, Tuple
from src.data_preprocessing.perturbations.utils.load_save import (
    load_data,
    save_data,
    INCLUDED_KEYS,
)
import argparse
import os

# Numerical entity labels
NUMERICAL_LABELS = {
    "CARDINAL",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
}


def replace_digits_with_new_sequence(original_text: str) -> str:
    """Replace digits in the original text with a new digit sequence.

    Generate a new digit sequence for the numerical entity, ensuring it doesn't
    start with zero and is different from the original.

    Args:
        original_text: The original text of the numerical entity.

    Returns:
        The perturbed text with digits replaced accordingly.
    """
    original_digits = "".join(c for c in original_text if c.isdigit())

    if not original_digits:
        return original_text  # No digits to replace

    while True:
        new_digits = []
        for i in range(len(original_digits)):
            if i == 0:
                new_digit = random.choice(string.digits[1:])  # No leading zero
            else:
                new_digit = random.choice(string.digits)
            new_digits.append(new_digit)
        new_digits_str = "".join(new_digits)

        if new_digits_str != original_digits:
            break  # Ensure the new digits are different

    new_text_list = list(original_text)
    digit_pos = 0
    for idx, char in enumerate(original_text):
        if char.isdigit():
            new_text_list[idx] = new_digits[digit_pos]
            digit_pos += 1

    return "".join(new_text_list)


def random_replacement(claim: str, entities: List[Dict[str, Any]]) -> str:
    """Randomly replace numerical values in the claim with random numbers.

    Args:
        claim: The original claim text.
        entities: List of entities with their positions.

    Returns:
        The claim with numerical values replaced by random numbers.
    """
    perturbed_claim = claim
    for entity in sorted(entities, key=lambda x: x["start_char"], reverse=True):
        if entity["label"] in NUMERICAL_LABELS:
            start = entity["start_char"]
            end = entity["end_char"]
            original_text = perturbed_claim[start:end]
            new_text = replace_digits_with_new_sequence(original_text)
            perturbed_claim = (
                perturbed_claim[:start] + new_text + perturbed_claim[end:]
            )
    return perturbed_claim


def masking(claim: str, entities: List[Dict[str, Any]]) -> str:
    """Mask numerical values in the claim.

    Replace digits, commas, and periods in the claim with '#' symbols.

    Args:
        claim: The original claim text.
        entities: List of entities with their positions.

    Returns:
        The claim with numerical values masked.
    """
    perturbed_claim = claim
    chars_to_mask = set(string.digits + ",.")
    for entity in sorted(entities, key=lambda x: x["start_char"], reverse=True):
        if entity["label"] in NUMERICAL_LABELS:
            start = entity["start_char"]
            end = entity["end_char"]
            original_text = perturbed_claim[start:end]
            new_text = "".join(
                "#" if char in chars_to_mask else char for char in original_text
            )
            perturbed_claim = (
                perturbed_claim[:start] + new_text + perturbed_claim[end:]
            )
    return perturbed_claim


def value_removal(claim: str, entities: List[Dict[str, Any]]) -> str:
    """Remove numerical values entirely from the claim.

    Args:
        claim: The original claim text.
        entities: List of entities with their positions.

    Returns:
        The claim with numerical values removed.
    """
    perturbed_claim = claim
    for entity in sorted(entities, key=lambda x: x["start_char"], reverse=True):
        if entity["label"] in NUMERICAL_LABELS:
            start = entity["start_char"]
            end = entity["end_char"]
            perturbed_claim = perturbed_claim[:start] + perturbed_claim[end:]
    perturbed_claim = " ".join(perturbed_claim.split())
    return perturbed_claim


def process_json(
    input_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns three separate lists for each perturbation type.

    Args:
        input_data: List of input data entries.

    Returns:
        A tuple containing:
        - List of entries with random replacements.
        - List of entries with masked values.
        - List of entries with removed values.
    """
    random_replacement_data = []
    masking_data = []
    value_removal_data = []

    for entry in input_data:
        claim: str = entry["normalized_claim"]
        entities: List[Dict[str, Any]] = entry.get("entities", [])

        perturbed_claim1 = random_replacement(claim, entities)
        perturbed_claim2 = masking(claim, entities)
        perturbed_claim3 = value_removal(claim, entities)

        base_entry = {key: entry[key] for key in INCLUDED_KEYS if key in entry}

        if perturbed_claim1 != claim:
            random_entry = base_entry.copy()
            random_entry["random_replacement_false"] = perturbed_claim1
            random_replacement_data.append(random_entry)

        if perturbed_claim2 != claim:
            masking_entry = base_entry.copy()
            masking_entry["masking_false"] = perturbed_claim2
            masking_data.append(masking_entry)

        if perturbed_claim3 != claim:
            removal_entry = base_entry.copy()
            removal_entry["value_removal_false"] = perturbed_claim3
            value_removal_data.append(removal_entry)

    return random_replacement_data, masking_data, value_removal_data


def main() -> None:
    """Main function to execute the obfuscation perturbation process."""
    parser = argparse.ArgumentParser(
        description="Apply obfuscation perturbations to claims."
    )
    parser.add_argument(
        "--input", "-i", type=str, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Directory to save output JSON files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_name = os.path.basename(args.input).split(".")[0]
    random_output = os.path.join(
        args.output_dir, f"{base_name}_random_replacement.json"
    )
    masking_output = os.path.join(args.output_dir, f"{base_name}_masking.json")
    removal_output = os.path.join(
        args.output_dir, f"{base_name}_value_removal.json"
    )

    data: List[Dict[str, Any]] = load_data(args.input)

    random_data, masking_data, removal_data = process_json(data)

    save_data(random_data, random_output)
    save_data(masking_data, masking_output)
    save_data(removal_data, removal_output)

    print(f"Random replacement data saved to: {random_output}")
    print(f"Masking data saved to: {masking_output}")
    print(f"Value removal data saved to: {removal_output}")


if __name__ == "__main__":
    main()
