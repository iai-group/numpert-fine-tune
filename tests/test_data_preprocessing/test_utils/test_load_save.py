"""Tests for the load_data and save_data functions in the load_save module."""

import os
import json
from tempfile import NamedTemporaryFile
from src.data_preprocessing.perturbations.utils.load_save import (
    load_data,
    save_data,
)


def test_load_data_list():
    """Test loading data from a JSON file with a list of dictionaries."""
    test_data = [{"key1": "value1"}, {"key2": "value2"}]
    with NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        json.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    loaded_data = load_data(temp_file_path)
    assert loaded_data == test_data

    os.remove(temp_file_path)


def test_load_data_dict():
    """Test loading data from a JSON file with a dictionary."""
    test_data = {"item1": {"key1": "value1"}, "item2": {"key2": "value2"}}
    expected_data = [{"key1": "value1"}, {"key2": "value2"}]
    with NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        json.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    loaded_data = load_data(temp_file_path)
    assert loaded_data == expected_data

    os.remove(temp_file_path)


def test_save_data():
    """Test saving data to a JSON file."""
    test_data = [{"key1": "value1"}, {"key2": "value2"}]
    with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name

    save_data(test_data, temp_file_path)

    with open(temp_file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        assert loaded_data == test_data

    os.remove(temp_file_path)
