"""Tests for the annotation orchestration."""

import json

from src.number_annotation.annotators.annotator import FakeAnnotator
from src.number_annotation.pipeline.run_annotation import annotate_record, annotate_split
from src.number_annotation.core.schema import RawNumber


def _raw(surface, context, **kw):
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


def _fake():
    """Build a FakeAnnotator covering one claim and one doc."""
    claim = "The tower is 330 meters tall."
    doc = "Records show the tower stands 330 meters in total."
    return (
        claim,
        doc,
        FakeAnnotator(
            {
                claim: [_raw("330 meters", "is 330 meters tall", value=330.0)],
                doc: [_raw("330 meters", "stands 330 meters in", value=330.0)],
            }
        ),
    )


class TestAnnotateRecord:
    """Tests for annotate_record."""

    def test_attaches_claim_and_evidence_numbers(self):
        """Verifies both sides are annotated and located."""
        claim, doc, annotator = _fake()
        record = {"claim_id": "x1", "claim": claim, "doc": doc}
        out = annotate_record(record, annotator)
        assert len(out["claim_numbers"]) == 1
        assert len(out["evidence_numbers"]) == 1
        c = out["claim_numbers"][0]
        assert claim[c["char_start"] : c["char_end"]] == "330 meters"
        e = out["evidence_numbers"][0]
        assert doc[e["char_start"] : e["char_end"]] == "330 meters"

    def test_validation_flags_present(self):
        """Verifies per-record validation flags are recorded."""
        claim, doc, annotator = _fake()
        record = {"claim_id": "x1", "claim": claim, "doc": doc}
        out = annotate_record(record, annotator)
        assert out["_annotation"]["claim_spans_valid"] is True
        assert out["_annotation"]["evidence_spans_valid"] is True

    def test_preserves_original_fields(self):
        """Verifies original record fields survive."""
        claim, doc, annotator = _fake()
        record = {
            "claim_id": "x1",
            "claim": claim,
            "doc": doc,
            "label": "True",
        }
        out = annotate_record(record, annotator)
        assert out["label"] == "True"
        assert out["claim_id"] == "x1"


class TestAnnotateSplit:
    """Tests for annotate_split + file IO."""

    def test_writes_output_file(self, tmp_path):
        """Verifies the split is annotated and written to disk."""
        claim, doc, annotator = _fake()
        data = [{"claim_id": "x1", "claim": claim, "doc": doc}]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        stats = annotate_split(str(in_path), str(out_path), annotator)

        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(result) == 1
        assert "evidence_numbers" in result[0]
        assert stats["records"] == 1
        assert stats["evidence_spans_valid"] == 1
