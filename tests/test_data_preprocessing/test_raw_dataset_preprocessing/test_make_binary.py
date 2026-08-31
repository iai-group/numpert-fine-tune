"""Derive the binary (True/False) splits from stripped QuanTemp data.

The legacy ``binary_data`` folder (deleted 2026-07-03) was verified to be
exactly QuanTemp minus the Conflicting-labeled records — with docs that the
old ``remove_referance.py`` had decimal-corrupted. The replacement derives
binary splits directly from the *stripped* QuanTemp output by label filter.
"""

from src.data_preprocessing.raw_dataset_preprocessing.make_binary import (
    make_binary_records,
)


def _qt(id, label, datasplit="test"):
    return {"id": id, "datasplit": datasplit, "label": label,
            "claim": f"claim {id}", "doc": f"doc {id}",
            "url": "https://example.com/x"}


def test_conflicting_records_dropped():
    records = [_qt(0, "True"), _qt(1, "Conflicting"), _qt(2, "False")]
    out = make_binary_records(records)
    assert [r["label"] for r in out] == ["True", "False"]


def test_claim_id_derived_from_datasplit_and_id():
    out = make_binary_records([_qt(7, "True", datasplit="train")])
    assert out[0]["claim_id"] == "train_000007"
    assert out[0]["id"] == 7  # original fields preserved


def test_input_not_mutated():
    records = [_qt(0, "True")]
    make_binary_records(records)
    assert "claim_id" not in records[0]
