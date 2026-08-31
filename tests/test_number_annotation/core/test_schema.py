"""Tests for the number annotation data model."""

from src.number_annotation.core.schema import (
    MODIFIERS,
    NumberSpan,
    RawNumber,
    SIGNS,
    VALUE_TYPES,
    raw_from_dict,
    span_from_dict,
)


class TestVocabularies:
    """Tests for the controlled vocabularies."""

    def test_value_types_expected(self):
        """Verifies the value_type vocabulary."""
        assert VALUE_TYPES == {
            "count",
            "percent",
            "money",
            "ordinal",
            "date_year",
            "duration",
            "fraction",
            "ratio",
            "other",
        }

    def test_modifiers_expected(self):
        """Verifies the modifier vocabulary."""
        assert MODIFIERS == {
            "exact",
            "approx",
            "at_least",
            "at_most",
            "range",
        }

    def test_signs_expected(self):
        """Verifies the sign vocabulary."""
        assert SIGNS == {"+", "-"}


class TestRawNumber:
    """Tests for the RawNumber dataclass."""

    def test_roundtrip(self):
        """Verifies RawNumber serializes and deserializes."""
        raw = RawNumber(
            surface="about 40%",
            value=40.0,
            value_low=None,
            value_high=None,
            value_type="percent",
            unit="%",
            modifier="approx",
            sign="+",
            context="reached about 40% of households",
        )
        assert raw_from_dict(raw.to_dict()) == raw


class TestNumberSpan:
    """Tests for the NumberSpan dataclass."""

    def test_from_raw_attaches_offsets_and_id(self):
        """Verifies a NumberSpan is built from a RawNumber + offsets."""
        raw = RawNumber(
            surface="1.2 million",
            value=1200000.0,
            value_low=None,
            value_high=None,
            value_type="count",
            unit="people",
            modifier="exact",
            sign="+",
            context="up from 1.2 million people",
        )
        span = NumberSpan.from_raw(raw, id=3, char_start=8, char_end=19)
        assert span.id == 3
        assert span.char_start == 8
        assert span.char_end == 19
        assert span.surface == "1.2 million"
        assert span.value == 1200000.0

    def test_roundtrip(self):
        """Verifies NumberSpan serializes and deserializes."""
        span = NumberSpan(
            id=0,
            surface="5",
            char_start=0,
            char_end=1,
            value=5.0,
            value_low=None,
            value_high=None,
            value_type="count",
            unit=None,
            modifier="exact",
            sign="+",
            context="5 apples",
        )
        assert span_from_dict(span.to_dict()) == span


def test_numberspan_context_fields_default_none():
    """New context fields default to None when unset."""
    from src.number_annotation.core.schema import NumberSpan

    span = NumberSpan(
        id=0,
        surface="5",
        char_start=0,
        char_end=1,
        value=5.0,
        value_low=None,
        value_high=None,
        value_type="count",
        unit=None,
        modifier="exact",
        sign="+",
        context="",
    )
    assert span.sentence_id is None
    assert span.sentence_start is None
    assert span.sentence_end is None
    assert span.context_sentence is None
    assert span.context_prev_curr is None
    assert span.context_window3 is None
    assert span.block_sentence_ids is None


def test_span_from_dict_reads_context_fields():
    """span_from_dict round-trips the new context fields."""
    from src.number_annotation.core.schema import span_from_dict

    data = {
        "id": 1,
        "surface": "5",
        "char_start": 7,
        "char_end": 8,
        "value": 5.0,
        "value_low": None,
        "value_high": None,
        "value_type": "count",
        "unit": None,
        "modifier": "exact",
        "sign": "+",
        "context": "",
        "sentence_id": 2,
        "sentence_start": 4,
        "sentence_end": 14,
        "context_sentence": "We had 5 cats.",
        "context_prev_curr": "Intro. We had 5 cats.",
        "context_window3": "Intro. We had 5 cats. Then more.",
        "block_sentence_ids": {
            "sentence": [2],
            "prev_curr": [1, 2],
            "window3": [1, 2, 3],
        },
    }
    span = span_from_dict(data)
    assert span.sentence_id == 2
    assert span.block_sentence_ids["window3"] == [1, 2, 3]
    assert span.context_sentence == "We had 5 cats."


def test_span_from_dict_without_context_fields_defaults_none():
    """Older dicts lacking context fields still deserialize."""
    from src.number_annotation.core.schema import span_from_dict

    data = {
        "id": 0,
        "surface": "5",
        "char_start": 0,
        "char_end": 1,
        "value": 5.0,
        "value_low": None,
        "value_high": None,
        "value_type": "count",
        "unit": None,
        "modifier": "exact",
        "sign": "+",
        "context": "",
    }
    span = span_from_dict(data)
    assert span.sentence_id is None
    assert span.block_sentence_ids is None
