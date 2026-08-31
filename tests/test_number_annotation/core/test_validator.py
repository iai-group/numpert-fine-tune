"""Tests for structural validation of located spans."""

from src.number_annotation.core.schema import NumberSpan
from src.number_annotation.core.validator import validate_spans


def _span(id, start, end, surface):
    """Build a minimal NumberSpan for tests."""
    return NumberSpan(
        id=id,
        surface=surface,
        char_start=start,
        char_end=end,
        value=None,
        value_low=None,
        value_high=None,
        value_type="count",
        unit=None,
        modifier="exact",
        sign="+",
        context="",
    )


class TestValidateSpans:
    """Tests for validate_spans."""

    def test_all_valid(self):
        """Verifies clean spans report ok."""
        text = "5 and 6"
        spans = [_span(0, 0, 1, "5"), _span(1, 6, 7, "6")]
        report = validate_spans(text, spans)
        assert report.ok is True
        assert report.mismatches == []
        assert report.overlaps == []

    def test_surface_mismatch_detected(self):
        """Verifies an offset that doesn't match surface is flagged."""
        text = "5 and 6"
        spans = [_span(0, 0, 1, "9")]  # text[0:1] == "5", not "9"
        report = validate_spans(text, spans)
        assert report.ok is False
        assert report.mismatches == [0]

    def test_overlap_detected(self):
        """Verifies overlapping spans are flagged."""
        text = "1234"
        spans = [_span(0, 0, 2, "12"), _span(1, 1, 3, "23")]
        report = validate_spans(text, spans)
        assert report.ok is False
        assert (0, 1) in report.overlaps

    def test_empty_is_valid(self):
        """Verifies empty span list is ok."""
        report = validate_spans("anything", [])
        assert report.ok is True
