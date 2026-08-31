"""Tests for the annotator parsing and implementations."""

from types import SimpleNamespace

from src.number_annotation.annotators.annotator import (
    ClaudeCodeAnnotator,
    DEFAULT_ANNOTATOR_MODEL,
    FakeAnnotator,
    parse_response,
)
from src.number_annotation.core.schema import RawNumber


def _payload():
    """Return a sample structured-output payload."""
    return {
        "numbers": [
            {
                "surface": "330 meters",
                "value": 330.0,
                "value_low": None,
                "value_high": None,
                "value_type": "count",
                "unit": "meters",
                "modifier": "exact",
                "sign": "+",
                "context": "is 330 meters tall",
            }
        ]
    }


class TestParseResponse:
    """Tests for parse_response."""

    def test_parses_one_number(self):
        """Verifies a single number is parsed into a RawNumber."""
        out = parse_response(_payload())
        assert out == [
            RawNumber(
                surface="330 meters",
                value=330.0,
                value_low=None,
                value_high=None,
                value_type="count",
                unit="meters",
                modifier="exact",
                sign="+",
                context="is 330 meters tall",
            )
        ]

    def test_empty_numbers(self):
        """Verifies an empty list is handled."""
        assert parse_response({"numbers": []}) == []

    def test_missing_numbers_key(self):
        """Verifies a missing 'numbers' key yields an empty list."""
        assert parse_response({}) == []


class TestFakeAnnotator:
    """Tests for the FakeAnnotator test double."""

    def test_returns_preprogrammed(self):
        """Verifies the fake returns mapped raw numbers."""
        raw = parse_response(_payload())
        fake = FakeAnnotator({"the tower": raw})
        assert fake.annotate("the tower") == raw

    def test_unknown_text_returns_empty(self):
        """Verifies unmapped text returns an empty list."""
        fake = FakeAnnotator({})
        assert fake.annotate("nothing here") == []


class TestClaudeCodeAnnotator:
    """Tests for ClaudeCodeAnnotator with an injected runner."""

    def test_default_model(self):
        """Verifies the default frontier model alias."""
        assert DEFAULT_ANNOTATOR_MODEL == "opus"

    def test_build_command_has_expected_flags(self):
        """Verifies the claude CLI command is well-formed."""
        annotator = ClaudeCodeAnnotator(model="opus")
        cmd = annotator.build_command()
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "--json-schema" in cmd
        assert cmd[cmd.index("--model") + 1] == "opus"

    def test_annotate_runs_and_parses_structured_output(self):
        """Verifies the runner is invoked and JSON is parsed."""
        import json

        captured = {}

        def fake_runner(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            payload = {"structured_output": _payload()}
            return SimpleNamespace(stdout=json.dumps(payload), returncode=0)

        annotator = ClaudeCodeAnnotator(model="opus", runner=fake_runner)
        out = annotator.annotate("The Eiffel Tower is 330 meters tall.")
        assert out[0].surface == "330 meters"
        # The document text reaches the CLI via stdin (input=).
        assert "330 meters tall" in captured["input"]
