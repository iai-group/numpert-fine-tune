"""Tests for annotation prompts and the structured-output schema."""

from src.number_annotation.core.prompts import (
    ANNOTATION_OUTPUT_SCHEMA,
    ANNOTATION_SYSTEM_PROMPT,
    build_annotation_user_prompt,
)
from src.number_annotation.core.schema import MODIFIERS, VALUE_TYPES


class TestSystemPrompt:
    """Tests for the system prompt."""

    def test_nonempty(self):
        """Verifies the system prompt is non-empty."""
        assert ANNOTATION_SYSTEM_PROMPT.strip()


class TestUserPrompt:
    """Tests for the user prompt builder."""

    def test_includes_text(self):
        """Verifies the input text is embedded verbatim."""
        text = "The Eiffel Tower is 330 meters tall."
        prompt = build_annotation_user_prompt(text)
        assert text in prompt


class TestOutputSchema:
    """Tests for the structured-output JSON schema."""

    def test_top_level_shape(self):
        """Verifies the schema requires a 'numbers' array."""
        assert ANNOTATION_OUTPUT_SCHEMA["type"] == "object"
        assert ANNOTATION_OUTPUT_SCHEMA["required"] == ["numbers"]
        assert ANNOTATION_OUTPUT_SCHEMA["additionalProperties"] is False

    def test_item_enums_match_vocabularies(self):
        """Verifies enum constraints match the schema vocabularies."""
        item = ANNOTATION_OUTPUT_SCHEMA["properties"]["numbers"]["items"]
        props = item["properties"]
        assert set(props["value_type"]["enum"]) == VALUE_TYPES
        assert set(props["modifier"]["enum"]) == MODIFIERS
        assert item["additionalProperties"] is False

    def test_item_required_fields(self):
        """Verifies every field is required (nullable via anyOf)."""
        item = ANNOTATION_OUTPUT_SCHEMA["properties"]["numbers"]["items"]
        assert set(item["required"]) == {
            "surface",
            "value",
            "value_low",
            "value_high",
            "value_type",
            "unit",
            "modifier",
            "sign",
            "context",
        }
