"""Numeration Perturbation."""

import re
from typing import Any, Dict, List, Optional, Set
from num2words import num2words  # type: ignore
from src.data_preprocessing.perturbations.utils.load_save import (
    INCLUDED_KEYS,
)

PERTUBATION_METHOD = "numeration"

# Numerical entity types
numerical_entities: Set[str] = {
    "CARDINAL",
    "DATE",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "TIME",
}

# Perturbation factors
perturbation_factors: Dict[str, float] = {
    "PERCENT": 0.1,  # 10%
    "MONEY": 0.1,  # 10%
    "CARDINAL": 0.1,  # 10%
    "DATE": 5,  # 5 days or years
    "TIME": 60,  # 60 minutes
}


def modify_number(number: float, entity_type: str, truth_value: str) -> float:
    """Modify a number based on its entity type and truth value.

    Args:
        number (float): The original number.
        entity_type (str): The type of numerical entity
            (e.g., "PERCENT", "MONEY").
        truth_value (str): Indicates whether to perturb the number
            ("false") or keep it ("true").

    Returns:
        float: The modified number.
    """
    if truth_value == "true":
        return number

    factor = perturbation_factors.get(entity_type, 1)
    original_number = (
        number  # Store the original number for accurate comparison
    )

    if entity_type in {"PERCENT", "MONEY", "CARDINAL"}:
        modified_number = number * (1 + factor)
        modified_number = round(modified_number)
        # Compare rounded integers to avoid floating-point precision issues
        if int(modified_number) == int(round(original_number)):
            modified_number += 1
        return modified_number
    elif entity_type in {"DATE", "TIME"}:
        return number + factor
    else:
        return number + factor


def numeration_perturbation(
    entity_text: str, entity_type: str, truth_value: str = "true"
) -> str:
    """Perturb the entity text by modifying the numbers.

    Args:
        entity_text (str): The original text of the numerical entity
            (e.g., "50 percent", "$50").
        entity_type (str): The type of numerical entity
            (e.g., "PERCENT", "MONEY").
        truth_value (str, optional): Indicates whether to
        perturb the number ("false") or keep it ("true"). Defaults to "true".

    Returns:
        str: The perturbed numerical entity text.
    """

    def replace_func(match: re.Match) -> str:
        try:
            if entity_type == "MONEY":
                number_str = match.group(2)  # Numerical digits
                unit = ""  # No additional unit for MONEY entities
            else:
                number_str = match.group(1)
                unit = (
                    match.group(2)
                    if match.lastindex is not None and match.lastindex >= 2
                    else ""
                )
            # Remove unwanted characters and convert to float
            clean_number_str = number_str.replace(",", "")
            number = float(clean_number_str)
            # Modify the number based on perturbation factors
            modified_number = modify_number(number, entity_type, truth_value)
            # Convert the modified number to words
            number_word = num2words(modified_number)
            # Handle singular and plural forms
            if entity_type == "MONEY":
                dollar_unit = "dollar" if modified_number == 1 else "dollars"
                return f"{number_word} {dollar_unit}"
            elif entity_type == "PERCENT":
                return f"{number_word} percent"
            else:
                return f"{number_word}{unit}"
        except Exception as e:
            # For debugging purposes
            print(f"Error in replace_func: {e}")
            return match.group(0)

    # Only handle digit-based numeration
    if any(char.isdigit() for char in entity_text):
        if entity_type == "MONEY":
            # Matches "$50", "€1,000", etc.
            pattern = r"([$€£]?)(\d+(?:,\d+)*(?:\.\d+)?)"
        elif entity_type == "PERCENT":
            # Matches "50%", "50 percent", etc.
            pattern = r"(\d+(?:,\d+)*(?:\.\d+)?)(%|\s*percent)?"
        else:
            # Matches general numbers possibly followed by a unit
            pattern = r"(\d+(?:,\d+)*(?:\.\d+)?)(\s*\w+)?"

        # Use the pattern to substitute matches using replace_func
        return re.sub(pattern, replace_func, entity_text, flags=re.IGNORECASE)
    else:
        # If no digits are present, return the original text
        return entity_text


def adjust_entity_bounds(
    sentence: str, entity: Dict[str, Any], start: int, end: int
) -> tuple[int, int]:
    """Adjust entity boundaries for special characters."""
    if entity["label"] == "MONEY" and start > 0:
        while start > 0 and sentence[start - 1] in "$€£":
            start -= 1
    if entity["label"] == "PERCENT" and end < len(sentence):
        while end < len(sentence) and sentence[end] == "%":
            end += 1
    return start, end


def apply_text_replacement(
    sentence: str, new_text: str, start: int, end: int
) -> str:
    """Replace text in sentence at given positions."""
    return sentence[:start] + new_text + sentence[end:]


def clean_sentence(sentence: str) -> str:
    """Clean up whitespace and punctuation spacing."""
    sentence = re.sub(r"\s+", " ", sentence)
    return re.sub(r"\s+([.,!?;:])", r"\1", sentence).strip()


def process_entity(
    entity: Dict[str, Any], truth_value: str, sentence: str
) -> tuple[str, bool]:
    """Process a single entity and return modified sentence.

    Changes the entity text if it is a numerical entity.
    """
    if entity["label"] not in numerical_entities:
        return sentence, False

    original_text = entity["text"]
    new_text = numeration_perturbation(
        original_text, entity["label"], truth_value
    )

    if new_text == original_text:
        return sentence, False

    start, end = adjust_entity_bounds(
        sentence, entity, entity["start_char"], entity["end_char"]
    )
    modified_sentence = apply_text_replacement(sentence, new_text, start, end)
    return modified_sentence, True


def process_sentence(
    content: Dict[str, Any], truth_value: str
) -> Optional[str]:
    """Process a single sentence to perturb its numerical entities."""
    sentence = content["normalized_claim"]
    entities = content.get("entities", [])
    perturbed_sentence = sentence
    perturbation_made = False

    # Sort entities in reverse order to avoid offset issues
    sorted_entities = sorted(
        entities, key=lambda e: e["start_char"], reverse=True
    )

    for entity in sorted_entities:
        perturbed_sentence, changed = process_entity(
            entity, truth_value, perturbed_sentence
        )
        perturbation_made = perturbation_made or changed

    if not perturbation_made:
        return None

    return clean_sentence(perturbed_sentence)


def process_json(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process the input JSON data to apply numeration perturbations.

    Only includes entries that have been altered.

    Args:
        input_data (List[Dict[str, Any]]): The input JSON data
        as a list of dictionaries.

    Returns:
        List[Dict[str, Any]]: The perturbed data in the specified
            output format, containing only altered claims.
    """
    output_data: List[Dict[str, Any]] = []
    for entry in input_data:
        label = entry.get("label", "UNKNOWN")

        # Apply perturbations for "true" and "false" truth values
        perturbed_type1 = process_sentence(entry, "true")
        perturbed_type2 = process_sentence(entry, "false")

        # Include only if at least one perturbation was made
        if perturbed_type1 or perturbed_type2:
            # Create a new entry with only the included keys
            perturbed_entry = {
                key: entry[key] for key in INCLUDED_KEYS if key in entry
            }
            perturbed_entry["pertubation_method"] = PERTUBATION_METHOD
            if perturbed_type1:
                perturbed_entry[f"perturb_type1_{label}"] = perturbed_type1
            if perturbed_type2:
                perturbed_entry[f"perturb_type2_{label}"] = perturbed_type2
            output_data.append(perturbed_entry)
    return output_data
