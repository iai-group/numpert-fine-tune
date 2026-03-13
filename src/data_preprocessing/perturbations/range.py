"""Range perturbation functions for numerical entities.

This module provides functions to perturb numerical entities in text by
converting them into range formats. It supports various entity types such as
percentages, monetary values, time durations, and more.
"""

import logging
import math
import re
from typing import Optional, Any, List, Dict

PERTURBATION_NAME = "range"

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

# Comprehensive set of currency symbols
currency_symbols = set("¢$€£¥₹₽₩₺₴₪₫₦₱₲₳₴₵₭₼R")
# Create a regex-safe string from the currency symbols set for use in regex
currency_regex_pattern = re.escape("".join(currency_symbols))


def extract_number_from_text(
    entity_text: str, entity_type: str
) -> tuple[str, Optional[str], str, str]:
    """Extract number and unit from the entity text based on the entity type.

    Args:
        entity_text: The text containing the numerical entity.
        entity_type: The type of the entity (e.g., "PERCENT", "MONEY").

    Returns:
        A tuple containing:
        - The prefix text before the number.
        - The extracted number as a string (or None if not found).
        - The suffix text after the number.
        - The unit associated with the number (e.g., "%", "hours").
    """
    if entity_type == "PERCENT":
        match = re.match(
            r"(.*?)([\d.,]+)\s*(%|percent)(.*)", entity_text, re.IGNORECASE
        )
        if match:
            prefix, number_text, unit, suffix = match.groups()
            return prefix, number_text, suffix, unit.strip()
    elif entity_type == "MONEY":
        match = re.match(
            rf"(.*?)([{currency_regex_pattern}]?)\s*([\d.,]+)(.*)", entity_text
        )
        if match:
            prefix, unit, number_text, suffix = match.groups()
            return prefix, number_text, suffix, unit.strip()
    elif entity_type in ["CARDINAL", "ORDINAL"]:
        match = re.match(r"(.*?)([\d,]+)(.*)", entity_text)
        if match:
            prefix, number_text, suffix = match.groups()
            return prefix, number_text, suffix, ""
    elif entity_type == "TIME":
        match = re.match(
            r"(.*?)([\d.,]+)\s*(hours?|minutes?|days?)(.*)",
            entity_text,
            re.IGNORECASE,
        )
        if match:
            prefix, number_text, unit, suffix = match.groups()
            return prefix, number_text, suffix, unit.strip()
    return "", None, "", ""


# Perturbation factors
perturbation_factors = {
    "PERCENT": 0.1,  # 10%
    "MONEY": 0.1,  # 10%
    "CARDINAL": 0.1,  # 10%
    "ORDINAL": 1,  # 1 rank
    "TIME": 0.1,  # 10%
}


def range_perturbation(  # noqa: C901
    entity_text: str, entity_type: str, truth_value: str = "true"
) -> str:
    """Perturb a numerical entity by converting it into a range format.

    Args:
        entity_text: The text containing the numerical entity.
        entity_type: The type of the entity (e.g., "PERCENT", "MONEY").
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed text with the numerical entity converted to a range.
    """
    prefix, number_text, suffix, unit = extract_number_from_text(
        entity_text, entity_type
    )
    if number_text is None:
        return entity_text

    try:
        perturbed = ""
        # --- Start of Generalized Fix ---
        # If the prefix itself is a currency symbol, always treat it as money.
        # This is a robust, general fix for any entity type.
        stripped_prefix = prefix.strip()
        if stripped_prefix in currency_symbols:
            money_unit = stripped_prefix
            # Remove the symbol from the prefix, preserving whitespace
            prefix = prefix.replace(money_unit, "", 1)
            perturbed = perturb_money(number_text, money_unit, truth_value)
        else:
            # If no currency prefix, proceed with normal entity type logic
            if entity_type == "TIME":
                perturbed = perturb_time(number_text, unit, truth_value)
            elif entity_type == "PERCENT":
                perturbed = perturb_percent(number_text, unit, truth_value)
            elif entity_type == "MONEY":
                perturbed = perturb_money(number_text, unit, truth_value)
            elif entity_type == "CARDINAL":
                perturbed = perturb_cardinal(number_text, truth_value)
            elif entity_type == "ORDINAL":
                perturbed = perturb_ordinal(number_text, truth_value)
            else:
                return entity_text
        # --- End of Generalized Fix ---

        # Final assembly: if there's a wordy prefix, ensure a
        # space before "between"
        if prefix.strip() and perturbed.startswith("between"):
            return f"{prefix.rstrip()} {perturbed}{suffix}"

        return f"{prefix}{perturbed}{suffix}"
    except Exception as e:
        logging.error(
            f"Error perturbing entity '{entity_text}' of type '{entity_type}': {e}"  # noqa: E501
        )
        return entity_text


def perturb_time(number_text: str, unit: str, truth_value: str) -> str:
    """Perturb a time entity by converting it into a range format.

    Args:
        number_text: The numerical value as a string.
        unit: The unit of time (e.g., "hours", "minutes").
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed time entity in range format.
    """
    num = float(number_text.replace(",", ""))
    unit = unit.lower()
    unit_mapping = {
        "minute": "minutes",
        "minutes": "minutes",
        "hour": "hours",
        "hours": "hours",
        "day": "days",
        "days": "days",
    }
    if unit not in unit_mapping:
        return f"{number_text} {unit}"
    delta = num * perturbation_factors["TIME"]
    lower, upper = (
        (max(num - delta, 0), num + delta)
        if truth_value == "true"
        else (num + 1.5 * delta, num + 2.5 * delta)
    )
    lower_formatted, upper_formatted = format_number(
        math.floor(lower), integer=True
    ), format_number(math.ceil(upper), integer=True)
    return (
        f"between {lower_formatted} and {upper_formatted} {unit}"
        if lower_formatted != upper_formatted
        else f"{lower_formatted} {unit}"
    )


def perturb_percent(number_text: str, unit: str, truth_value: str) -> str:
    """Perturb a percentage entity by converting it into a range format.

    Args:
        number_text: The numerical value as a string.
        unit: The unit of percentage (e.g., "%").
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed percentage entity in range format.
    """
    num = float(number_text.replace(",", ""))
    delta = num * perturbation_factors["PERCENT"]
    lower, upper = (
        (max(num - delta, 0), num + delta)
        if truth_value == "true"
        else (num * 1.5, num * 1.6)
    )
    lower_formatted, upper_formatted = format_number(
        math.floor(lower)
    ), format_number(math.ceil(upper))
    unit_str = f" {unit}" if unit and not unit.startswith("%") else unit
    if lower_formatted == upper_formatted:
        return f"{lower_formatted}{unit_str}"
    return (
        f"between {lower_formatted}{unit_str} and {upper_formatted}{unit_str}"
    )


def perturb_money(number_text: str, unit: str, truth_value: str) -> str:
    """Perturb a monetary entity by converting it into a range format.

    Args:
        number_text: The numerical value as a string.
        unit: The unit of currency (e.g., "$", "€").
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed monetary entity in range format.
    """
    num = float(number_text.replace(",", ""))
    delta = num * perturbation_factors["MONEY"]
    lower, upper = (
        (max(num - delta, 0), num + delta)
        if truth_value == "true"
        else (num + 1.5 * delta, num + 2.5 * delta)
    )
    lower_formatted = format_currency(math.floor(lower), unit)
    upper_formatted = format_currency(math.ceil(upper), unit)
    return (
        f"between {lower_formatted} and {upper_formatted}"
        if lower_formatted != upper_formatted
        else lower_formatted
    )


def perturb_cardinal(number_text: str, truth_value: str) -> str:
    """Perturb a cardinal number entity by converting it into a range format.

    Args:
        number_text: The numerical value as a string.
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed cardinal number entity in range format.
    """
    num = float(number_text.replace(",", ""))
    delta = num * perturbation_factors["CARDINAL"]
    lower, upper = (
        (max(num - delta, 0), num + delta)
        if truth_value == "true"
        else (num * 1.45, num * 1.65)
    )
    lower_formatted = format_number(math.floor(lower), integer=True)
    upper_formatted = format_number(math.ceil(upper), integer=True)
    return (
        f"between {lower_formatted} and {upper_formatted}"
        if lower_formatted != upper_formatted
        else lower_formatted
    )


def perturb_ordinal(number_text: str, truth_value: str) -> str:
    """Perturb an ordinal number entity by converting it into a range format.

    Args:
        number_text: The numerical value as a string.
        truth_value: The truth value ("true" or "false") for perturbation.

    Returns:
        The perturbed ordinal number entity in range format.
    """
    match = re.match(r"(\d+)", number_text)
    if not match:
        return number_text
    num = int(match.group(1))
    delta = perturbation_factors["ORDINAL"]
    lower, upper = (
        (max(num - delta, 1), num + delta)
        if truth_value == "true"
        else (num + delta + 1, num + 2 * delta + 1)
    )
    lower_ordinal, upper_ordinal = get_ordinal(int(lower)), get_ordinal(
        int(upper)
    )
    return (
        f"between {lower_ordinal} and {upper_ordinal}"
        if lower_ordinal != upper_ordinal
        else lower_ordinal
    )


def get_ordinal(n: int) -> str:
    """Convert an integer to its ordinal representation.

    Args:
        n: The integer to convert.

    Returns:
        The ordinal representation of the integer (e.g., "1st", "2nd").
    """
    suffix = (
        "th"
        if 10 <= n % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def _is_integer_number(n: float | int) -> bool:
    """Return True when n is an integer value.

    Works for both int and float without relying on float-only APIs.
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def format_number(n: float | int, integer: bool = False) -> str:
    """Format a number as a string with comma separation and decimal places.

    Args:
        n: The number to format.
        integer: Whether to format as an integer (default: False).

    Returns:
        The formatted number as a string.
    """
    return f"{int(n):,}" if integer or _is_integer_number(n) else f"{n:,.1f}"


def format_currency(n: float | int, unit: str) -> str:
    """Format a currency value as a string with the unit and comma separation.

    Args:
        n: The numerical value of the currency.
        unit: The currency unit as a string (e.g., "$", "€").

    Returns:
        The formatted currency value as a string.
    """
    unit_str = f"{unit}" if unit else ""
    return (
        f"{unit_str}{int(n):,}"
        if _is_integer_number(n)
        else f"{unit_str}{n:,.2f}"
    )


def perturb_claim(
    claim_text: str, entities: List[Dict[str, Any]], truth_value: str
) -> str:
    """Perturbs all entities in a claim based on the truth value."""
    new_claim = claim_text
    entities_sorted = sorted(
        entities, key=lambda x: x["start_char"], reverse=True
    )

    for entity in entities_sorted:
        entity_text = entity["text"]
        entity_label = entity["label"]
        start_char = entity["start_char"]
        end_char = entity["end_char"]

        # --- Start of Context-Aware Fix ---
        # Check if a currency symbol exists right before the entity_text in the
        # main claim.
        # This handles cases where NER splits the symbol from the number
        # (e.g., "$" and "500").
        if start_char > 0 and claim_text[start_char - 1] in currency_symbols:
            # Prepend the symbol to the entity text to be processed.
            entity_text = claim_text[start_char - 1] + entity_text
            # Adjust the start character to ensure the symbol is replaced.
            start_char -= 1
        # --- End of Context-Aware Fix ---

        perturbed_text = range_perturbation(
            entity_text, entity_label, truth_value
        )

        if start_char == 0 and perturbed_text.lower().startswith("between"):
            perturbed_text = perturbed_text.capitalize()

        new_claim = (
            new_claim[:start_char] + perturbed_text + new_claim[end_char:]
        )
    return new_claim


def process_json(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes the input data and generates perturbed claims."""
    output_data: List[Dict[str, Any]] = []
    for entry in input_data:
        claim_text = entry.get("normalized_claim", "")
        entities = entry.get("entities", [])
        label = entry.get("label", "UNKNOWN")
        perturbed_entry = dict(entry)
        perturbed_entry["perturbation"] = PERTURBATION_NAME

        perturbed_entry[f"perturb_type1_{label}"] = perturb_claim(
            claim_text, entities, "true"
        )
        perturbed_entry[f"perturb_type2_{label}"] = perturb_claim(
            claim_text, entities, "false"
        )

        for key_to_remove in ["ner_tags", "tokens", "entities"]:
            perturbed_entry.pop(key_to_remove, None)

        output_data.append(perturbed_entry)
    return output_data
