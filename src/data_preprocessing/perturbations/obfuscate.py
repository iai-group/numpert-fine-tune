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
        return original_text

    while True:
        new_digits = []
        for i in range(len(original_digits)):
            if i == 0:
                new_digit = random.choice(string.digits[1:])
            else:
                new_digit = random.choice(string.digits)
            new_digits.append(new_digit)
        if "".join(new_digits) != original_digits:
            break

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
            new_text = replace_digits_with_new_sequence(
                perturbed_claim[start:end]
            )
            perturbed_claim = (
                perturbed_claim[:start] + new_text + perturbed_claim[end:]
            )
    return perturbed_claim


def masking(claim: str, entities: List[Dict[str, Any]]) -> str:
    """Mask numerical values in the claim with '#' symbols.

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
            new_text = "".join(
                "#" if char in chars_to_mask else char
                for char in perturbed_claim[start:end]
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
    return " ".join(perturbed_claim.split())


def process_json(
    input_data: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Returns a dict with 'true' and 'false' splits, each entry containing all
    applicable obfuscation perturbation types as keys.

    Args:
        input_data: List of input data entries.

    Returns:
        Dict with gold label keys mapping to lists of perturbed entries.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    for entry in input_data:
        claim: str = entry["normalized_claim"]
        entities: List[Dict[str, Any]] = entry.get("entities", [])
        label: str = str(entry.get("label", "UNKNOWN"))

        perturbed_random = random_replacement(claim, entities)
        perturbed_masked = masking(claim, entities)
        perturbed_removed = value_removal(claim, entities)

        if all(
            p == claim
            for p in [perturbed_random, perturbed_masked, perturbed_removed]
        ):
            continue

        base_entry = {key: entry[key] for key in INCLUDED_KEYS if key in entry}

        if perturbed_random != claim:
            base_entry["perturb_type3_random"] = perturbed_random
        if perturbed_masked != claim:
            base_entry["perturb_type3_masked"] = perturbed_masked
        if perturbed_removed != claim:
            base_entry["perturb_type3_removed"] = perturbed_removed

        results.setdefault(label, []).append(base_entry)

    return results
