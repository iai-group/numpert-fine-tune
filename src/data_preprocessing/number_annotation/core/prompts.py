"""Prompts and structured-output schema for number annotation."""

from src.number_annotation.core.schema import MODIFIERS, VALUE_TYPES

ANNOTATION_SYSTEM_PROMPT = (
    "You are a precise annotator of numeric mentions in text. "
    "Find EVERY explicit numeric mention: digits, spelled-out "
    "numbers, percentages, money, ordinals, years/dates, "
    "durations, fractions, ratios, magnitudes (e.g. '3 million'), "
    "and bounded or approximate forms (e.g. 'about 40', "
    "'more than 50', 'between 2 and 3'). Treat a full numeric "
    "expression as ONE mention. For each, copy the exact surface "
    "substring, a normalized numeric value, the value_type, the "
    "unit if any, the modifier, the sign, and a short surrounding "
    "context snippet copied verbatim from the text. Do not invent "
    "numbers that are not present. Do not rewrite the text."
)


def _nullable(type_name: str) -> dict:
    """Return an anyOf schema allowing the given type or null."""
    return {"anyOf": [{"type": type_name}, {"type": "null"}]}


ANNOTATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "value": _nullable("number"),
                    "value_low": _nullable("number"),
                    "value_high": _nullable("number"),
                    "value_type": {
                        "type": "string",
                        "enum": sorted(VALUE_TYPES),
                    },
                    "unit": _nullable("string"),
                    "modifier": {
                        "type": "string",
                        "enum": sorted(MODIFIERS),
                    },
                    "sign": {"type": "string", "enum": ["+", "-"]},
                    "context": {"type": "string"},
                },
                "required": [
                    "surface",
                    "value",
                    "value_low",
                    "value_high",
                    "value_type",
                    "unit",
                    "modifier",
                    "sign",
                    "context",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["numbers"],
    "additionalProperties": False,
}

_USER_TEMPLATE = (
    "Annotate every numeric mention in the text below.\n"
    "For a single exact value, set value and leave value_low/"
    "value_high null. For a range, set value_low and value_high "
    "and leave value null. unit is null when there is no unit.\n\n"
    "TEXT:\n{text}"
)


def build_annotation_user_prompt(text: str) -> str:
    """Build the user prompt embedding the text to annotate."""
    return _USER_TEMPLATE.format(text=text)
