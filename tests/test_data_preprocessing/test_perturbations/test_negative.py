"""Tests for the negative perturbation."""

from unittest.mock import patch

from src.data_preprocessing.perturbations.negative import (
    negative_number_perturbation,
    process_json,
)


def test_negative_number_perturbation_positive_integer():
    """Test that a positive integer percentage is made negative."""
    claim = "The success rate is 50%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 21}]
    expected_output = "The success rate is -50%."
    assert negative_number_perturbation(claim, entities) == expected_output


def test_negative_number_perturbation_already_negative():
    """Test that an already negative percentage is left unchanged."""
    claim = "The success rate is -50%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 22}]
    expected_output = "The success rate is -50%."
    assert negative_number_perturbation(claim, entities) == expected_output


def test_negative_number_perturbation_float():
    """Test that a positive float percentage is made negative."""
    claim = "The success rate is 50.5%."
    entities = [{"label": "PERCENT", "start_char": 18, "end_char": 23}]
    expected_output = "The success rate is -50.5%."
    assert negative_number_perturbation(claim, entities) == expected_output


def test_negative_number_perturbation_no_percent_entity():
    """Test that non-PERCENT entities are not perturbed."""
    claim = "There are 100 apples."
    entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 13}]
    assert negative_number_perturbation(claim, entities) == claim


def test_negative_number_perturbation_empty_entities():
    """Test that a claim with no entities is returned unchanged."""
    claim = "The success rate is 50%."
    entities = []
    assert negative_number_perturbation(claim, entities) == claim


def test_negative_number_perturbation_multiple_percent_entities():
    """Test that multiple PERCENT entities are all made negative."""
    claim = "Rates are 30% and 70%."
    entities = [
        {"label": "PERCENT", "start_char": 10, "end_char": 13},
        {"label": "PERCENT", "start_char": 18, "end_char": 21},
    ]
    result = negative_number_perturbation(claim, entities)
    assert "-30%" in result
    assert "-70%" in result


def test_negative_number_perturbation_no_number_in_entity():
    """Test that a PERCENT entity without a number is returned unchanged."""
    claim = "The rate is about half."
    entities = [{"label": "PERCENT", "start_char": 11, "end_char": 22}]
    assert negative_number_perturbation(claim, entities) == claim


def test_process_json_basic():
    """Test process_json with entries that should be perturbed."""
    input_data = [
        {
            "normalized_claim": "The success rate is 50%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
            "label": "True",
            "claim_id": "1",
        }
    ]
    with patch(
        "src.data_preprocessing.perturbations.negative.INCLUDED_KEYS",
        ["claim_id"],
    ):
        result = process_json(input_data)
    assert len(result) == 1
    assert result[0]["perturbation"] == "negative"
    assert result[0]["perturb_type2_True"] == "The success rate is -50%."
    assert result[0]["claim_id"] == "1"


def test_process_json_excludes_unperturbed():
    """Test that process_json excludes entries without changes."""
    input_data = [
        {
            "normalized_claim": "The company had no percentage.",
            "entities": [],
            "label": "True",
        }
    ]
    with patch(
        "src.data_preprocessing.perturbations.negative.INCLUDED_KEYS", []
    ):
        result = process_json(input_data)
    assert result == []


def test_process_json_unknown_label():
    """Test process_json uses UNKNOWN when label is missing."""
    input_data = [
        {
            "normalized_claim": "The success rate is 50%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
        }
    ]
    with patch(
        "src.data_preprocessing.perturbations.negative.INCLUDED_KEYS", []
    ):
        result = process_json(input_data)
    assert len(result) == 1
    assert result[0]["perturbation"] == "negative"
    assert "perturb_type2_UNKNOWN" in result[0]


def test_process_json_multiple_entries():
    """Test process_json with multiple entries, some perturbed and some not."""
    input_data = [
        {
            "normalized_claim": "The success rate is 50%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
            "label": "True",
        },
        {
            "normalized_claim": "No numbers here.",
            "entities": [],
            "label": "False",
        },
        {
            "normalized_claim": "Accuracy increased by 10%.",
            "entities": [
                {"label": "PERCENT", "start_char": 21, "end_char": 24}
            ],
            "label": "False",
        },
    ]
    with patch(
        "src.data_preprocessing.perturbations.negative.INCLUDED_KEYS", []
    ):
        result = process_json(input_data)
    assert len(result) == 2
    assert all(entry["perturbation"] == "negative" for entry in result)
    assert result[0]["perturb_type2_True"] == "The success rate is -50%."
    assert result[1]["perturb_type2_False"] == "Accuracy increased by -10%."


def test_process_json_included_keys_filtering():
    """Test that process_json only includes keys from INCLUDED_KEYS."""
    input_data = [
        {
            "normalized_claim": "The success rate is 50%.",
            "entities": [
                {"label": "PERCENT", "start_char": 18, "end_char": 21}
            ],
            "label": "True",
            "claim_id": "abc",
            "extra_field": "should_be_excluded",
        }
    ]
    with patch(
        "src.data_preprocessing.perturbations.negative.INCLUDED_KEYS",
        ["claim_id"],
    ):
        result = process_json(input_data)
    assert "claim_id" in result[0]
    assert "extra_field" not in result[0]
    assert "normalized_claim" not in result[0]
