"""Tests for the negative perturbation."""

import json

from src.data_preprocessing.perturbations.negative import (
    load_data,
    save_data,
    negative_number_perturbation,
    process_json,
)


def test_negative_number_perturbation():
    """Test the negative number perturbation function."""
    claim = "The success rate is 50%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 21}]
    expected_output = "The success rate is -50%."
    assert negative_number_perturbation(claim, entities) == expected_output

    claim = "The success rate is -50%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 22}]
    expected_output = "The success rate is -50%."
    assert negative_number_perturbation(claim, entities) == expected_output

    claim = "The success rate is 50.5%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 23}]
    expected_output = "The success rate is -50.5%."
    assert negative_number_perturbation(claim, entities) == expected_output


def test_process_json():
    """Test the process_json function."""
    input_data = {
        "url1": {
            "claim": "The success rate is 50%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
        },
        "url2": {
            "claim": "The success rate is 70%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
        },
    }
    expected_output = {
        "url1": {"negative_number_false": "The success rate is -50%."},
        "url2": {"negative_number_false": "The success rate is -70%."},
    }
    assert process_json(input_data) == expected_output


def test_load_data(tmp_path):
    """Test the load_data function."""
    data = {"key": "value"}
    file_path = tmp_path / "test.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_data = load_data(file_path)
    assert loaded_data == data


def test_save_data(tmp_path):
    """Test the save_data function."""
    data = {"key": "value"}
    file_path = tmp_path / "test.json"
    save_data(data, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data == data
