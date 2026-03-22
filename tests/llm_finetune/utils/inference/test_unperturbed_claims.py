"""Tests for unperturbed-claims deduplication helper."""

from datasets import Dataset

from src.llm_finetune.utils.inference.unperturbed_claims import (
    deduplicate_unperturbed_claims,
)


def _make_entry(claim_id, source_file="file_a.json"):
    """Minimal dataset row with the fields used by dedup logic."""
    return {
        "claim_id": claim_id,
        "perturbation_type": "original_normalized",
        "label": '{"label": true}',
        "conversations": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Claim: {claim_id}"},
            {"role": "assistant", "content": '{"label": true}'},
        ],
        "source_file": source_file,
    }


class TestDeduplicateUnperturbedClaims:
    """Test deduplicate_unperturbed_claims helper."""

    def test_no_duplicates(self):
        """Dataset with unique claim_ids passes through unchanged."""
        ds = Dataset.from_list(
            [_make_entry("c1"), _make_entry("c2"), _make_entry("c3")]
        )
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 3
        assert result["claim_id"] == ["c1", "c2", "c3"]

    def test_removes_duplicates(self):
        """Duplicate claim_ids (from different files) are removed."""
        ds = Dataset.from_list(
            [
                _make_entry("c1", "file_a.json"),
                _make_entry("c1", "file_b.json"),
                _make_entry("c2", "file_a.json"),
                _make_entry("c2", "file_b.json"),
                _make_entry("c3", "file_a.json"),
            ]
        )
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 3
        assert result["claim_id"] == ["c1", "c2", "c3"]

    def test_keeps_first_occurrence(self):
        """The first occurrence per claim_id is retained."""
        ds = Dataset.from_list(
            [
                _make_entry("c1", "first.json"),
                _make_entry("c1", "second.json"),
            ]
        )
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 1
        assert result[0]["source_file"] == "first.json"

    def test_empty_dataset(self):
        """Empty dataset returns empty."""
        ds = Dataset.from_list([])
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 0

    def test_single_entry(self):
        """Single-entry dataset is unchanged."""
        ds = Dataset.from_list([_make_entry("c1")])
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 1
        assert result[0]["claim_id"] == "c1"

    def test_many_duplicates(self):
        """Same claim_id repeated many times yields one entry."""
        entries = [_make_entry("c1", f"file_{i}.json") for i in range(10)]
        ds = Dataset.from_list(entries)
        result = deduplicate_unperturbed_claims(ds)
        assert len(result) == 1
        assert result[0]["source_file"] == "file_0.json"

    def test_preserves_order(self):
        """Claim ordering follows first-seen order."""
        ds = Dataset.from_list(
            [
                _make_entry("c3"),
                _make_entry("c1"),
                _make_entry("c3"),
                _make_entry("c2"),
                _make_entry("c1"),
            ]
        )
        result = deduplicate_unperturbed_claims(ds)
        assert result["claim_id"] == ["c3", "c1", "c2"]
