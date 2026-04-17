"""Tests for src.llm_finetune.utils.inference.resume."""

import json
import pytest

from src.llm_finetune.utils.inference.resume import load_resume_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ENTRY = {
    "index": 0,
    "claim_id": "c001",
    "perturbation_type": "original_normalized",
    "label": "True",
    "full_prompt": "prompt text",
    "raw_response": '{"label": true}',
    "parsed_successfully": True,
    "predicted_label": '{"label": true}',
}


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# load_resume_data – happy path
# ---------------------------------------------------------------------------


class TestLoadResumeDataHappyPath:
    def test_loads_single_valid_entry(self, tmp_path):
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [_VALID_ENTRY])

        raw_outputs, completed = load_resume_data(str(p), num_examples=5)

        assert 0 in completed
        assert len(completed) == 1
        assert raw_outputs[0] == _VALID_ENTRY
        assert raw_outputs[1] == {}  # un-filled slots remain empty dicts

    def test_loads_multiple_valid_entries(self, tmp_path):
        entries = [
            {**_VALID_ENTRY, "index": i, "claim_id": f"c{i:03d}"}
            for i in range(3)
        ]
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, entries)

        raw_outputs, completed = load_resume_data(str(p), num_examples=5)

        assert completed == {0, 1, 2}
        for i in range(3):
            assert raw_outputs[i]["claim_id"] == f"c{i:03d}"

    def test_returns_empty_when_file_is_blank(self, tmp_path):
        p = tmp_path / "details.jsonl"
        p.write_text("", encoding="utf-8")

        raw_outputs, completed = load_resume_data(str(p), num_examples=3)

        assert completed == set()
        assert all(v == {} for v in raw_outputs)

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "details.jsonl"
        p.write_text("\n" + json.dumps(_VALID_ENTRY) + "\n\n", encoding="utf-8")

        _, completed = load_resume_data(str(p), num_examples=5)

        assert 0 in completed


# ---------------------------------------------------------------------------
# load_resume_data – schema validation
# ---------------------------------------------------------------------------


class TestLoadResumeDataSchemaValidation:
    def test_skips_invalid_json_lines(self, tmp_path):
        p = tmp_path / "details.jsonl"
        p.write_text(
            "not json\n" + json.dumps(_VALID_ENTRY) + "\n", encoding="utf-8"
        )

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == {0}

    def test_skips_non_dict_entry(self, tmp_path):
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [[1, 2, 3]])  # list, not dict

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == set()

    @pytest.mark.parametrize(
        "missing_field",
        [
            "index",
            "claim_id",
            "perturbation_type",
            "label",
            "full_prompt",
            "raw_response",
            "parsed_successfully",
            "predicted_label",
        ],
    )
    def test_skips_entry_missing_required_field(self, tmp_path, missing_field):
        entry = {k: v for k, v in _VALID_ENTRY.items() if k != missing_field}
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [entry])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == set()

    def test_skips_entry_with_non_int_index(self, tmp_path):
        entry = {**_VALID_ENTRY, "index": "0"}  # string, not int
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [entry])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == set()

    def test_skips_entry_with_negative_index(self, tmp_path):
        entry = {**_VALID_ENTRY, "index": -1}
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [entry])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == set()

    def test_skips_entry_with_out_of_bounds_index(self, tmp_path):
        entry = {**_VALID_ENTRY, "index": 5}  # num_examples=5, so 5 is OOB
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [entry])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == set()

    def test_accepts_null_predicted_label(self, tmp_path):
        """predicted_label=None is valid (parse failure that was recorded)."""
        entry = {**_VALID_ENTRY, "predicted_label": None}
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [entry])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert 0 in completed

    def test_partial_and_full_entries_mixed(self, tmp_path):
        """Only fully-valid entries are loaded; partial ones are skipped."""
        partial = {k: v for k, v in _VALID_ENTRY.items() if k != "full_prompt"}
        full = {**_VALID_ENTRY, "index": 1, "claim_id": "c001b"}
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [partial, full])

        _, completed = load_resume_data(str(p), num_examples=5)

        assert completed == {1}

    def test_raw_outputs_length_equals_num_examples(self, tmp_path):
        p = tmp_path / "details.jsonl"
        _write_jsonl(p, [_VALID_ENTRY])

        raw_outputs, _ = load_resume_data(str(p), num_examples=7)

        assert len(raw_outputs) == 7
