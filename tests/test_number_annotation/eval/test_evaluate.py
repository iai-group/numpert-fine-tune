"""Tests for annotation evaluation."""

from src.number_annotation.eval.evaluate import (
    entities_to_spans,
    score_spans,
)
from src.number_annotation.core.schema import NumberSpan


def _span(id, start, end, value=None, value_type="count"):
    """Build a NumberSpan for tests."""
    return NumberSpan(
        id=id,
        surface="x",
        char_start=start,
        char_end=end,
        value=value,
        value_low=None,
        value_high=None,
        value_type=value_type,
        unit=None,
        modifier="exact",
        sign="+",
        context="",
    )


class TestScoreSpans:
    """Tests for score_spans."""

    def test_perfect_match(self):
        """Verifies identical spans score 1.0 across the board."""
        gold = [_span(0, 0, 3, value=5.0)]
        pred = [_span(0, 0, 3, value=5.0)]
        s = score_spans(pred, gold)
        assert s.exact_f1 == 1.0
        assert s.overlap_f1 == 1.0
        assert s.value_accuracy == 1.0
        assert s.type_accuracy == 1.0

    def test_overlap_but_not_exact(self):
        """Verifies overlap match counts for overlap, not exact."""
        gold = [_span(0, 0, 5)]
        pred = [_span(0, 2, 7)]  # overlaps but offsets differ
        s = score_spans(pred, gold)
        assert s.exact_f1 == 0.0
        assert s.overlap_f1 == 1.0

    def test_missing_and_spurious(self):
        """Verifies recall/precision penalize misses and extras."""
        gold = [_span(0, 0, 3), _span(1, 10, 13)]
        pred = [_span(0, 0, 3), _span(1, 20, 23)]  # one wrong
        s = score_spans(pred, gold)
        assert s.exact_precision == 0.5
        assert s.exact_recall == 0.5

    def test_value_accuracy_on_matched(self):
        """Verifies value accuracy is over matched pairs only."""
        gold = [_span(0, 0, 3, value=5.0)]
        pred = [_span(0, 0, 3, value=9.0)]  # matched offset, wrong value
        s = score_spans(pred, gold)
        assert s.exact_f1 == 1.0
        assert s.value_accuracy == 0.0

    def test_empty_gold_and_pred(self):
        """Verifies empty inputs score 1.0 (vacuously correct)."""
        s = score_spans([], [])
        assert s.exact_f1 == 1.0
        assert s.overlap_f1 == 1.0


class TestEntitiesToSpans:
    """Tests for the spaCy-entity baseline converter."""

    def test_maps_label_and_offsets(self):
        """Verifies entity dicts become NumberSpans."""
        entities = [
            {
                "text": "330",
                "label": "CARDINAL",
                "start_char": 13,
                "end_char": 16,
            }
        ]
        spans = entities_to_spans(entities)
        assert len(spans) == 1
        assert spans[0].char_start == 13
        assert spans[0].char_end == 16
        assert spans[0].value_type == "count"
        assert spans[0].value == 330.0

    def test_unparseable_value_is_none(self):
        """Verifies non-numeric surfaces yield value None."""
        entities = [
            {
                "text": "today",
                "label": "DATE",
                "start_char": 0,
                "end_char": 5,
            }
        ]
        spans = entities_to_spans(entities)
        assert spans[0].value is None
        assert spans[0].value_type == "date_year"
