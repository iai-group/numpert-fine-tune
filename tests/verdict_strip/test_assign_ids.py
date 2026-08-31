"""Tests for Step 0 id/datasplit assignment."""

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.assign_ids import (  # noqa: E501
    assign_ids,
)


def test_assigns_position_id_and_datasplit():
    recs = [{"doc": "a"}, {"doc": "b"}, {"doc": "c"}]
    out = assign_ids(recs, "test")
    assert [r["id"] for r in out] == [0, 1, 2]
    assert all(r["datasplit"] == "test" for r in out)


def test_idempotent_keeps_existing_id():
    recs = [{"doc": "a", "id": 7, "datasplit": "train"}, {"doc": "b"}]
    out = assign_ids(recs, "val")
    # existing id/datasplit preserved; only the record missing id gets one
    assert out[0]["id"] == 7 and out[0]["datasplit"] == "train"
    assert out[1]["id"] == 1 and out[1]["datasplit"] == "val"


def test_does_not_reorder_or_drop():
    recs = [{"doc": str(i)} for i in range(5)]
    out = assign_ids(recs, "test")
    assert [r["doc"] for r in out] == ["0", "1", "2", "3", "4"]
