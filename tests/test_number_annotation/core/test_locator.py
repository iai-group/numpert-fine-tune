"""Tests for deterministic offset resolution."""

from src.number_annotation.core.locator import (
    ALL_OCCURRENCES_CLAIMED,
    SURFACE_NOT_FOUND,
    find_occurrences,
    locate_numbers,
)
from src.number_annotation.core.schema import RawNumber


def _raw(surface, context="", **kw):
    """Build a minimal RawNumber for tests."""
    return RawNumber(
        surface=surface,
        value=kw.get("value"),
        value_low=None,
        value_high=None,
        value_type=kw.get("value_type", "count"),
        unit=None,
        modifier="exact",
        sign="+",
        context=context,
    )


class TestFindOccurrences:
    """Tests for find_occurrences."""

    def test_single(self):
        """Verifies a single occurrence is found."""
        assert find_occurrences("there are 5 apples", "5") == [10]

    def test_multiple(self):
        """Verifies all occurrences are found."""
        assert find_occurrences("5 and 5 and 5", "5") == [0, 6, 12]

    def test_none(self):
        """Verifies missing surface yields no occurrences."""
        assert find_occurrences("no digits", "5") == []


class TestLocateNumbers:
    """Tests for locate_numbers."""

    def test_single_number(self):
        """Verifies one number is located with correct offsets."""
        text = "The tower is 330 meters tall."
        spans, failures = locate_numbers(text, [_raw("330 meters")])
        assert failures == []
        assert len(spans) == 1
        assert text[spans[0].char_start : spans[0].char_end] == "330 meters"
        assert spans[0].id == 0

    def test_duplicate_surface_uses_context(self):
        """Verifies context disambiguates duplicate surfaces."""
        text = "We had 5 cats. Later we had 5 dogs."
        raw = [
            _raw("5", context="had 5 dogs"),
            _raw("5", context="had 5 cats"),
        ]
        spans, failures = locate_numbers(text, raw)
        assert failures == []
        # Each '5' mapped to a distinct, non-overlapping offset.
        starts = sorted(s.char_start for s in spans)
        assert starts == [7, 28]

    def test_unfindable_surface_reported(self):
        """Verifies a surface not in the text becomes a failure."""
        text = "no numbers here"
        spans, failures = locate_numbers(text, [_raw("42")])
        assert spans == []
        assert len(failures) == 1
        assert failures[0].surface == "42"

    def test_absent_surface_reason_is_surface_not_found(self):
        """Verifies an absent surface gets reason surface_not_found."""
        _, failures = locate_numbers("no numbers here", [_raw("42")])
        assert failures[0].reason == SURFACE_NOT_FOUND
        assert failures[0].duplicate_of is None

    def test_duplicate_emission_reports_claimed_and_dup_flag(self):
        """Verifies a re-emitted surface fails as a flagged duplicate."""
        text = "the rate was 75% last year"
        raw = [
            _raw("75%", context="rate was 75% last"),
            _raw("75%", context="rate was 75% last"),
        ]
        spans, failures = locate_numbers(text, raw)
        # First emission located; second cannot find a free slot.
        assert len(spans) == 1
        assert len(failures) == 1
        assert failures[0].reason == ALL_OCCURRENCES_CLAIMED
        assert failures[0].duplicate_of == spans[0].id

    def test_ids_are_sequential_over_located(self):
        """Verifies located spans get sequential ids."""
        text = "1 and 2"
        spans, failures = locate_numbers(text, [_raw("1"), _raw("2")])
        assert [s.id for s in spans] == [0, 1]
        assert failures == []
