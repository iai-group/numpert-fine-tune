"""Tests for the spaCy-track orchestration (offline, FakeLocator)."""

import json

import pytest

from src.number_annotation.pipeline.run_annotation_spacy import (
    _resolve_io,
    annotate_record,
    annotate_split,
)
from src.number_annotation.core.schema import NumberSpan


class FakeLocator:
    """Deterministic locator mapping exact text to located spans."""

    def __init__(self, mapping):
        self._mapping = mapping

    def locate(self, text):
        return list(self._mapping.get(text, []))


def _span(char_start, char_end, surface):
    return NumberSpan(
        id=0,
        surface=surface,
        char_start=char_start,
        char_end=char_end,
        value=None,
        value_low=None,
        value_high=None,
        value_type="count",
        unit=None,
        modifier="exact",
        sign="+",
        context="",
    )


def test_annotate_record_shape_and_enrichment():
    """Record gains claim_numbers, evidence_numbers, _annotation."""
    doc = "We had 5 cats. Then 6 left."
    five = doc.index("5")
    locator = FakeLocator({doc: [_span(five, five + 1, "5")]})
    record = {"claim_id": "x", "claim": "", "doc": doc}

    out = annotate_record(record, locator)

    assert out["claim_id"] == "x"
    assert out["claim_numbers"] == []
    assert len(out["evidence_numbers"]) == 1
    span = out["evidence_numbers"][0]
    # Enrichment applied:
    assert span["context_sentence"].strip() == "We had 5 cats."
    assert span["sentence_id"] is not None
    # Annotation block present with parity fields:
    ann = out["_annotation"]
    assert ann["evidence_spans_valid"] is True
    assert ann["evidence_failures"] == 0
    assert ann["evidence_failure_details"] == []


def test_annotate_record_handles_missing_doc():
    """A record without 'doc' yields empty evidence_numbers."""
    locator = FakeLocator({})
    out = annotate_record({"claim_id": "y"}, locator)
    assert out["evidence_numbers"] == []
    assert out["_annotation"]["evidence_spans_valid"] is True


class TestResolveIo:
    """Tests for _resolve_io (input/output resolution)."""

    def test_split_resolves_binary_convention(self):
        """A --split maps to the binary_rederived input/output paths."""
        inp, out = _resolve_io(None, None, "test")
        assert inp == "data/raw_data/verdict_stripped/binary_rederived/test_binary.json"
        assert out == "data/spacy_number_annotated/test_spacy.json"

    def test_explicit_input_requires_output(self):
        """--input without --output is an error."""
        with pytest.raises(ValueError):
            _resolve_io("some/in.json", None, None)

    def test_explicit_input_and_output(self):
        """--input/--output are returned unchanged."""
        inp, out = _resolve_io("in.json", "out.json", None)
        assert (inp, out) == ("in.json", "out.json")

    def test_requires_exactly_one_of_input_or_split(self):
        """Neither or both of input/split is an error."""
        with pytest.raises(ValueError):
            _resolve_io(None, None, None)
        with pytest.raises(ValueError):
            _resolve_io("in.json", "out.json", "test")


class TestClaimIdAssignment:
    """Tests for claim_id back-fill in annotate_split."""

    def test_assigns_claim_id_when_missing(self, tmp_path):
        """Records lacking claim_id get a stable positional id."""
        data = [{"claim": "c0", "doc": "d0"}, {"claim": "c1", "doc": "d1"}]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        annotate_split(str(in_path), str(out_path), FakeLocator({}))

        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert [r["claim_id"] for r in result] == [
            "test_000000",
            "test_000001",
        ]

    def test_preserves_existing_claim_id(self, tmp_path):
        """An existing claim_id is left untouched."""
        data = [{"claim_id": "kept_1", "claim": "c", "doc": "d"}]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        annotate_split(str(in_path), str(out_path), FakeLocator({}))

        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result[0]["claim_id"] == "kept_1"


class TestEnrichmentAndAnnotation:
    """Located evidence spans are enriched and the block is spaCy-shaped."""

    def test_evidence_spans_get_per_window_block_ids(self):
        """Each evidence span carries the per-window context + id dict."""
        doc = "One came. We had 5 cats. Then 6 left."
        five = doc.index("5")
        locator = FakeLocator({doc: [_span(five, five + 1, "5")]})
        out = annotate_record(
            {"claim_id": "x", "claim": "", "doc": doc}, locator
        )
        s = out["evidence_numbers"][0]
        assert s["context_sentence"].strip() == "We had 5 cats."
        assert "One came." in s["context_prev_curr"]
        assert "Then 6 left." in s["context_window3"]
        assert set(s["block_sentence_ids"]) == {
            "sentence",
            "prev_curr",
            "window3",
        }
        assert s["block_sentence_ids"]["sentence"] == [1]
        assert s["block_sentence_ids"]["window3"] == [0, 1, 2]

    def test_two_numbers_same_sentence_share_block_ids(self):
        """Numbers in one sentence share per-window ids (groupable/dedupe)."""
        doc = "We had 5 cats and 7 dogs."
        five, seven = doc.index("5"), doc.index("7")
        locator = FakeLocator(
            {doc: [_span(five, five + 1, "5"), _span(seven, seven + 1, "7")]}
        )
        out = annotate_record(
            {"claim_id": "x", "claim": "", "doc": doc}, locator
        )
        a, b = out["evidence_numbers"]
        assert a["block_sentence_ids"] == b["block_sentence_ids"]

    def test_claim_is_annotated_independently(self):
        """The claim is annotated into claim_numbers."""
        claim = "Up 5 percent."
        five = claim.index("5")
        locator = FakeLocator({claim: [_span(five, five + 1, "5")]})
        out = annotate_record(
            {"claim_id": "x", "claim": claim, "doc": ""}, locator
        )
        assert len(out["claim_numbers"]) == 1
        assert out["claim_numbers"][0]["surface"] == "5"
        assert out["_annotation"]["claim_spans_valid"] is True

    def test_spacy_track_always_reports_zero_failures(self):
        """spaCy offsets are exact, so failure counts/details are empty."""
        doc = "Has 5 cats."
        five = doc.index("5")
        out = annotate_record(
            {"claim_id": "x", "claim": "", "doc": doc},
            FakeLocator({doc: [_span(five, five + 1, "5")]}),
        )
        ann = out["_annotation"]
        assert ann["claim_failures"] == 0
        assert ann["evidence_failures"] == 0
        assert ann["claim_failure_details"] == []
        assert ann["evidence_failure_details"] == []

    def test_preserves_original_record_fields(self):
        """Original metadata fields survive annotation."""
        out = annotate_record(
            {
                "claim_id": "x",
                "claim": "",
                "doc": "",
                "label": "Conflicting",
                "url": "http://e",
            },
            FakeLocator({}),
        )
        assert out["label"] == "Conflicting"
        assert out["url"] == "http://e"

    def test_missing_claim_and_doc_yield_empty_valid(self):
        """A record with no claim/doc annotates to empty but valid."""
        out = annotate_record({"claim_id": "x"}, FakeLocator({}))
        assert out["claim_numbers"] == []
        assert out["evidence_numbers"] == []
        assert out["_annotation"]["claim_spans_valid"] is True
        assert out["_annotation"]["evidence_spans_valid"] is True


class TestAnnotateSplitStats:
    """annotate_split aggregates validity and writes the corpus."""

    def test_stats_and_file_written(self, tmp_path):
        """Stats count records + valid sides; output is round-trippable."""
        doc = "Has 5 cats."
        five = doc.index("5")
        data = [
            {"claim_id": "a", "claim": "", "doc": doc},
            {"claim_id": "b", "claim": "", "doc": ""},
        ]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        stats = annotate_split(
            str(in_path),
            str(out_path),
            FakeLocator({doc: [_span(five, five + 1, "5")]}),
        )

        assert stats == {
            "records": 2,
            "claim_spans_valid": 2,
            "evidence_spans_valid": 2,
        }
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(result) == 2
        assert len(result[0]["evidence_numbers"]) == 1
        assert result[1]["evidence_numbers"] == []
