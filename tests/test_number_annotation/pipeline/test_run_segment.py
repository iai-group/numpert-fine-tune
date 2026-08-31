"""Tests for sentence-segmentation dataset prep."""

import json

import pytest

from src.number_annotation.pipeline.run_segment import (
    _resolve_input,
    build_parser,
    segment_dataset,
    segment_record,
)


class TestSegmentRecord:
    """Tests for segment_record."""

    def test_adds_offset_complete_sentences(self):
        """doc_sentences slice the doc exactly and ids are contiguous."""
        doc = "The tower is tall. It is in Paris."
        out = segment_record({"claim_id": "x1", "doc": doc})
        sents = out["doc_sentences"]
        assert len(sents) == 2
        assert [s["id"] for s in sents] == [0, 1]
        for s in sents:
            assert doc[s["start"]:s["end"]] == s["text"]

    def test_preserves_original_fields_and_leaves_claim_whole(self):
        """All original fields survive; claim is not segmented."""
        record = {
            "claim_id": "x1",
            "claim": "A claim. With two sentences.",
            "doc": "One. Two.",
            "label": "True",
        }
        out = segment_record(record)
        assert out["claim"] == "A claim. With two sentences."
        assert out["label"] == "True"
        assert out["claim_id"] == "x1"
        assert "claim_sentences" not in out

    def test_empty_doc_yields_no_sentences(self):
        """An empty/whitespace doc yields an empty sentence list."""
        out = segment_record({"claim_id": "x1", "doc": "   \n "})
        assert out["doc_sentences"] == []

    def test_missing_doc_key_yields_no_sentences(self):
        """A record with no doc key yields an empty sentence list."""
        out = segment_record({"claim_id": "x1"})
        assert out["doc_sentences"] == []
        assert out["claim_id"] == "x1"

    def test_does_not_mutate_input(self):
        """The input record is not mutated in place."""
        record = {"claim_id": "x1", "doc": "One. Two."}
        segment_record(record)
        assert "doc_sentences" not in record


class TestSegmentDataset:
    """Tests for segment_dataset + file IO."""

    def test_writes_output_and_reports_stats(self, tmp_path):
        """Segments every record, writes JSON, returns correct stats."""
        data = [
            {"claim_id": "x1", "doc": "One. Two."},
            {"claim_id": "x2", "doc": ""},
        ]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "nested" / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        stats = segment_dataset(str(in_path), str(out_path))

        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(result) == 2
        assert len(result[0]["doc_sentences"]) == 2
        assert result[1]["doc_sentences"] == []
        assert stats == {
            "records": 2,
            "sentences_total": 2,
            "empty_docs": 1,
        }

    def test_limit_truncates_records(self, tmp_path):
        """--limit processes only the first N records."""
        data = [{"claim_id": f"x{i}", "doc": "One. Two."} for i in range(3)]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        in_path.write_text(json.dumps(data), encoding="utf-8")

        stats = segment_dataset(str(in_path), str(out_path), limit=2)

        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(result) == 2
        assert stats["records"] == 2


class TestCli:
    """Tests for input resolution and argument parsing."""

    def test_resolve_input_from_explicit_path(self):
        """An explicit --input path is returned unchanged."""
        assert _resolve_input("some/file.json", None) == "some/file.json"

    def test_resolve_input_from_split(self):
        """A --split resolves to the binary_rederived path."""
        assert (
            _resolve_input(None, "test")
            == "data/raw_data/verdict_stripped/binary_rederived/test_binary.json"
        )

    def test_resolve_input_requires_exactly_one(self):
        """Neither or both of input/split is an error."""
        with pytest.raises(ValueError):
            _resolve_input(None, None)
        with pytest.raises(ValueError):
            _resolve_input("some/file.json", "test")

    def test_parser_accepts_expected_args(self):
        """The parser exposes --input, --split, --output, --limit."""
        parser = build_parser()
        args = parser.parse_args(
            ["--input", "in.json", "--output", "out.json", "--limit", "5"]
        )
        assert args.input == "in.json"
        assert args.output == "out.json"
        assert args.limit == 5
        assert args.split is None
