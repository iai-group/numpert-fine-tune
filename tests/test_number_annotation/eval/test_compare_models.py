"""Tests for the model-comparison helper (offline)."""

from src.number_annotation.eval.compare_models import compare, summarize


def _span(start, end, value=None, value_type="count"):
    """Build a span dict for tests."""
    return {
        "id": 0,
        "surface": "x",
        "char_start": start,
        "char_end": end,
        "value": value,
        "value_low": None,
        "value_high": None,
        "value_type": value_type,
        "unit": None,
        "modifier": "exact",
        "sign": "+",
        "context": "",
    }


def _rec(claim_id, evidence, ev_fail=0):
    """Build an annotated record dict for tests."""
    return {
        "claim_id": claim_id,
        "claim_numbers": [],
        "evidence_numbers": evidence,
        "_annotation": {
            "claim_spans_valid": True,
            "evidence_spans_valid": True,
            "claim_failures": 0,
            "evidence_failures": ev_fail,
        },
    }


class TestSummarize:
    """Tests for summarize."""

    def test_counts(self):
        """Verifies located and failure counts aggregate correctly."""
        records = [
            _rec("a", [_span(0, 3), _span(5, 8)]),
            _rec("b", [_span(0, 3)], ev_fail=1),
        ]
        out = summarize(records)
        assert out["records"] == 2
        assert out["evidence_numbers"] == 3
        assert out["evidence_failures"] == 1
        assert out["evidence_spans_valid_records"] == 2


class TestCompare:
    """Tests for compare."""

    def test_identical_is_perfect(self):
        """Verifies identical spans score 1.0 agreement."""
        ref = [_rec("a", [_span(0, 3, value=5.0)])]
        cand = [_rec("a", [_span(0, 3, value=5.0)])]
        out = compare(ref, cand)
        assert out["shared"] == 1
        assert out["overlap_f1"] == 1.0
        assert out["value_accuracy"] == 1.0

    def test_missing_span_lowers_recall(self):
        """Verifies a candidate miss reduces agreement below 1.0."""
        ref = [_rec("a", [_span(0, 3), _span(10, 13)])]
        cand = [_rec("a", [_span(0, 3)])]
        out = compare(ref, cand)
        assert out["overlap_f1"] < 1.0

    def test_no_shared_records(self):
        """Verifies disjoint claim_ids report zero shared."""
        out = compare([_rec("a", [])], [_rec("b", [])])
        assert out == {"shared": 0}
