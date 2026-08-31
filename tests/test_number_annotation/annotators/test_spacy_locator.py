"""Tests for the spaCy locator transform and value parsing."""

import pytest
import spacy

from src.number_annotation.annotators.spacy_locator import (
    SpacyNumberLocator,
    ents_to_spans,
    parse_value,
)


class TestParseValue:
    """Tests for parse_value."""

    def test_plain_integer_with_comma(self):
        """'1,200' parses to 1200."""
        assert parse_value("1,200") == 1200.0

    def test_decimal(self):
        """'2.8' parses to 2.8."""
        assert parse_value("2.8") == 2.8

    def test_magnitude_word(self):
        """'2.8 million' parses to 2.8e6."""
        assert parse_value("2.8 million") == 2_800_000.0

    def test_percent_sign_stripped(self):
        """'45%' parses to 45."""
        assert parse_value("45%") == 45.0

    def test_currency_sign_stripped(self):
        """'$5' parses to 5."""
        assert parse_value("$5") == 5.0

    def test_spelled_out(self):
        """'three' parses to 3 via word2number."""
        assert parse_value("three") == 3.0

    def test_unparseable_returns_none(self):
        """A non-numeric token returns None."""
        assert parse_value("the") is None

    @pytest.mark.parametrize(
        "surface, expected",
        [
            ("5 thousand", 5_000.0),
            ("2 billion", 2_000_000_000.0),
            ("1.5 trillion", 1_500_000_000_000.0),
            ("$3.2 million", 3_200_000.0),
            ("5 MILLION", 5_000_000.0),  # magnitude is case-insensitive
            ("1,234,567", 1_234_567.0),
            ("-5", -5.0),
            ("-3.2 million", -3_200_000.0),
            ("£10", 10.0),
            ("€20", 20.0),
            ("2.8%", 2.8),
            ("about 1,200", 1_200.0),  # spaCy may include a leading hedge
            ("twenty-one", 21.0),
            ("one hundred", 100.0),
        ],
    )
    def test_numeric_surfaces(self, surface, expected):
        """A range of numeric surfaces parse to their float value."""
        assert parse_value(surface) == expected

    @pytest.mark.parametrize(
        "surface", ["", "   ", "the", "N/A", "third", "million"]
    )
    def test_non_numeric_surfaces_return_none(self, surface):
        """Empty/garbage/bare-magnitude surfaces parse to None."""
        assert parse_value(surface) is None


class _Ent:
    """Duck-typed stand-in for a spaCy entity span."""

    def __init__(self, text, label_, start_char, end_char):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class TestEntsToSpans:
    """Tests for ents_to_spans."""

    def test_numeric_entity_becomes_span_with_offsets(self):
        """A CARDINAL entity yields a span with copied offsets."""
        ents = [_Ent("330", "CARDINAL", 13, 16)]
        spans = ents_to_spans(ents)
        assert len(spans) == 1
        sp = spans[0]
        assert sp.char_start == 13
        assert sp.char_end == 16
        assert sp.surface == "330"
        assert sp.value == 330.0
        assert sp.value_type == "count"
        assert sp.modifier == "exact"
        assert sp.sign == "+"
        assert sp.unit is None

    def test_non_numeric_label_filtered_out(self):
        """A PERSON entity is ignored."""
        ents = [_Ent("Alice", "PERSON", 0, 5)]
        assert ents_to_spans(ents) == []

    def test_label_maps_to_value_type(self):
        """MONEY/PERCENT/DATE map to their value_types."""
        ents = [
            _Ent("$5", "MONEY", 0, 2),
            _Ent("45%", "PERCENT", 3, 6),
            _Ent("2012", "DATE", 7, 11),
        ]
        spans = ents_to_spans(ents)
        assert [s.value_type for s in spans] == [
            "money",
            "percent",
            "date_year",
        ]

    def test_ids_are_sequential(self):
        """Kept spans get sequential ids over kept entities only."""
        ents = [
            _Ent("1", "CARDINAL", 0, 1),
            _Ent("Bob", "PERSON", 2, 5),
            _Ent("2", "CARDINAL", 6, 7),
        ]
        spans = ents_to_spans(ents)
        assert [s.id for s in spans] == [0, 1]

    def test_all_numeric_labels_map(self):
        """Every kept NER label maps to its value_type."""
        ents = [
            _Ent("3", "CARDINAL", 0, 1),
            _Ent("3 kg", "QUANTITY", 2, 6),
            _Ent("$5", "MONEY", 7, 9),
            _Ent("45%", "PERCENT", 10, 13),
            _Ent("third", "ORDINAL", 14, 19),
            _Ent("2012", "DATE", 20, 24),
            _Ent("two hours", "TIME", 25, 34),
        ]
        spans = ents_to_spans(ents)
        assert [s.value_type for s in spans] == [
            "count",
            "count",
            "money",
            "percent",
            "ordinal",
            "date_year",
            "duration",
        ]

    def test_empty_entities_yield_no_spans(self):
        """No entities yields an empty span list."""
        assert ents_to_spans([]) == []

    def test_unparseable_numeric_entity_kept_with_none_value(self):
        """A numeric-labelled but unparseable surface is kept, value None."""
        spans = ents_to_spans([_Ent("several", "CARDINAL", 0, 7)])
        assert len(spans) == 1
        assert spans[0].value is None
        assert spans[0].value_type == "count"

    def test_offsets_and_defaults_copied_for_each(self):
        """Offsets are copied verbatim; rich fields take schema defaults."""
        ents = [_Ent("5", "CARDINAL", 4, 5), _Ent("$9", "MONEY", 9, 11)]
        spans = ents_to_spans(ents)
        assert [(s.char_start, s.char_end) for s in spans] == [(4, 5), (9, 11)]
        for s in spans:
            assert s.modifier == "exact"
            assert s.sign == "+"
            assert s.unit is None
            assert s.context == ""
            assert s.value_low is None
            assert s.value_high is None


class _FakeDoc:
    """Duck-typed spaCy Doc exposing `.ents`."""

    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    """Callable stand-in for a loaded spaCy pipeline."""

    def __init__(self, ents):
        self._ents = ents
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return _FakeDoc(self._ents)


class TestSpacyNumberLocator:
    """Tests for SpacyNumberLocator (model injected; never loads real trf)."""

    def test_empty_text_returns_empty_without_loading(self):
        """Empty text short-circuits before the model is ever loaded."""
        locator = SpacyNumberLocator()
        assert locator.locate("") == []
        assert locator._nlp is None

    def test_locate_converts_model_entities(self, monkeypatch):
        """locate() returns ents_to_spans of the model's entities."""
        ents = [_Ent("5", "CARDINAL", 7, 8), _Ent("Bob", "PERSON", 0, 3)]
        monkeypatch.setattr(spacy, "load", lambda name: _FakeNlp(ents))
        locator = SpacyNumberLocator(model="dummy")
        spans = locator.locate("Bob had 5 cats.")
        assert [s.surface for s in spans] == ["5"]  # PERSON filtered out

    def test_model_is_loaded_once_and_cached(self, monkeypatch):
        """The model loads lazily on first locate and is reused after."""
        loads = []
        fake = _FakeNlp([_Ent("5", "CARDINAL", 0, 1)])
        monkeypatch.setattr(
            spacy, "load", lambda name: loads.append(name) or fake
        )
        locator = SpacyNumberLocator(model="dummy")
        locator.locate("5 a")
        locator.locate("5 b")
        assert loads == ["dummy"]  # loaded exactly once
        assert fake.calls == 2  # but invoked per locate

    def test_use_gpu_requires_gpu_before_load(self, monkeypatch):
        """use_gpu=True activates the GPU before loading the model."""
        events = []
        monkeypatch.setattr(
            spacy, "require_gpu", lambda: events.append("gpu")
        )
        monkeypatch.setattr(
            spacy, "load", lambda name: events.append("load") or _FakeNlp([])
        )
        SpacyNumberLocator(model="dummy", use_gpu=True).locate("text 5")
        assert events == ["gpu", "load"]

    def test_no_gpu_does_not_require_gpu(self, monkeypatch):
        """The default (CPU) path never calls require_gpu."""
        called = []
        monkeypatch.setattr(
            spacy, "require_gpu", lambda: called.append("gpu")
        )
        monkeypatch.setattr(spacy, "load", lambda name: _FakeNlp([]))
        SpacyNumberLocator(model="dummy").locate("text 5")
        assert called == []
