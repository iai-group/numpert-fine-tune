"""Tests for the create_binary_dataset.py module."""

import json
import pytest
from src.data_preprocessing.raw_dataset_preprocessing.create_binary_dataset import (  # noqa: E501
    remove_label,
)


def test_remove_label_basic():
    """Test removing a specific label from a JSON dataset."""
    input_data = json.dumps(
        [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "B"},
            {"id": 3, "label": "A"},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, "A")
    expected = json.dumps([{"id": 1, "label": "B"}], indent=4)
    assert filtered_json == expected
    assert dropped == 2
    assert remaining == 1


def test_remove_label_no_removal():
    """Test when no entries are removed from the dataset."""
    input_data = json.dumps(
        [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "B"},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, "C")
    expected = json.dumps(
        [
            {"id": 0, "label": "A"},
            {"id": 1, "label": "B"},
        ],
        indent=4,
    )
    assert filtered_json == expected
    assert dropped == 0
    assert remaining == 2


def test_remove_label_all_removed():
    """Test when all entries are removed from the dataset."""
    input_data = json.dumps(
        [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, "A")
    expected = json.dumps([], indent=4)
    assert filtered_json == expected
    assert dropped == 2
    assert remaining == 0


def test_remove_label_empty_input():
    """Test removing a label from an empty JSON dataset."""
    input_data = json.dumps([])
    filtered_json, dropped, remaining = remove_label(input_data, "A")
    expected = json.dumps([], indent=4)
    assert filtered_json == expected
    assert dropped == 0
    assert remaining == 0


def test_remove_label_missing_label_key():
    """Test removing a label when some entries do not have the label key."""
    input_data = json.dumps(
        [
            {"id": 1},
            {"id": 2, "label": "B"},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, "B")
    expected = json.dumps([{"id": 0}], indent=4)
    assert filtered_json == expected
    assert dropped == 1
    assert remaining == 1


def test_remove_label_mixed_entries():
    """Test removing a label from a dataset with mixed entries."""
    input_data = json.dumps(
        [
            {"id": 1, "label": "A"},
            {"id": 2},
            {"id": 3, "label": "B"},
            {"id": 4, "label": "A"},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, "A")
    expected = json.dumps(
        [
            {"id": 1},
            {"id": 2, "label": "B"},
        ],
        indent=4,
    )
    assert filtered_json == expected
    assert dropped == 2
    assert remaining == 2


def test_remove_label_non_string_label():
    """Test removing a label when the label is not a string."""
    input_data = json.dumps(
        [
            {"id": 1, "label": 123},
            {"id": 2, "label": "B"},
            {"id": 3, "label": 123},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, 123)
    expected = json.dumps([{"id": 1, "label": "B"}], indent=4)
    assert filtered_json == expected
    assert dropped == 2
    assert remaining == 1


def test_remove_label_nested_json():
    """Test removing a label from a dataset with nested JSON objects."""
    nested = {"type": "A"}
    input_data = json.dumps(
        [
            {"id": 1, "label": nested},
            {"id": 2, "label": "B"},
            {"id": 3, "label": nested},
        ]
    )
    filtered_json, dropped, remaining = remove_label(input_data, nested)
    expected = json.dumps([{"id": 1, "label": "B"}], indent=4)
    assert filtered_json == expected
    assert dropped == 2
    assert remaining == 1


def test_remove_label_invalid_json():
    """Test handling invalid JSON input."""
    with pytest.raises(json.JSONDecodeError):
        remove_label("not a json", "A")
