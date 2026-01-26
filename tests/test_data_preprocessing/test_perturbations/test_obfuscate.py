"""Tests for obfuscation perturbation methods with failsafe."""

from unittest.mock import patch
from code.data_preprocessing.perturbations.obfuscate import (
    replace_digits_with_new_sequence,
    random_replacement,
    masking,
    value_removal,
    process_json,
)

# Existing Tests


def test_replace_digits_with_new_sequence():
    """Test replace_digits_with_new_sequence function.

    Tests three scenarios:
    1. String with only digits
    2. String with mixed content
    3. String with no digits
    """
    # Mock random.choice to return deterministic digits
    with patch("random.choice", side_effect=["4", "5", "6"]):
        # Test with only digits
        result = replace_digits_with_new_sequence("123")
        assert result == "456"  # '1'->'4', '2'->'5', '3'->'6'

    # Mock random.choice to return different deterministic digits
    with patch("random.choice", side_effect=["7", "8", "9"]):
        # Test with mixed content
        result = replace_digits_with_new_sequence("test123test")
        assert result == "test789test"

    # Test with no digits
    result = replace_digits_with_new_sequence("test")
    assert result == "test"


def test_random_replacement():
    """Test random_replacement function.

    Verifies that the function replaces numeric entities in the claim.
    """
    # Mock replace_digits_with_new_sequence to return a known value
    with patch(
        "code.data_preprocessing.perturbations.obfuscate."
        "replace_digits_with_new_sequence",
        return_value="456",
    ):
        claim = "There were 123 people"
        entities = [{"start_char": 11, "end_char": 14, "label": "CARDINAL"}]
        result = random_replacement(claim, entities)
        assert result == "There were 456 people"


def test_masking():
    """Test masking function.

    Verifies that numeric entities are replaced with '#' characters.
    """
    claim = "The price is $100"
    entities = [{"start_char": 13, "end_char": 17, "label": "MONEY"}]
    result = masking(claim, entities)
    assert result == "The price is $###"


def test_value_removal():
    """Test value_removal function.

    Verifies that numeric entities are removed from the claim.
    """
    claim = "Temperature: 23.5 degrees"
    entities = [{"start_char": 13, "end_char": 17, "label": "QUANTITY"}]
    result = value_removal(claim, entities)
    assert result == "Temperature: degrees"


def test_process_json_all_perturbations_alter():
    """Test process_json when all perturbations alter the claim.

    Verifies that all perturbed claims are included in the output.
    """
    input_data = {
        "url_1": {
            "claim": "Value is 100",
            "entities": [
                {"start_char": 9, "end_char": 12, "label": "CARDINAL"}
            ],
        }
    }
    # Mock perturbation functions to alter the claim deterministically
    with (
        patch(
            "code.data_preprocessing.perturbations.obfuscate.random_replacement",  # noqa: E501
            return_value="Value is 200",
        ),
        patch(
            "code.data_preprocessing.perturbations.obfuscate.masking",
            return_value="Value is ###",
        ),
        patch(
            "code.data_preprocessing.perturbations.obfuscate.value_removal",
            return_value="Value is",
        ),
    ):

        result = process_json(input_data)

        assert "url_1" in result
        assert result["url_1"]["random_replacement_false"] == "Value is 200"
        assert result["url_1"]["masking_false"] == "Value is ###"
        assert result["url_1"]["value_removal_false"] == "Value is"


def test_process_json_some_perturbations_alter():
    """Test process_json when some perturbations alter the claim.

    Verifies that only altered perturbations are included.
    """
    input_data = {
        "url_2": {
            "claim": "No numbers here",
            "entities": [
                {
                    "start_char": 0,
                    "end_char": 3,
                    "label": "NON_NUM",
                }  # Non-numerical label
            ],
        }
    }
    # Mock perturbation functions; since the label is
    # non-numerical, perturbations should not alter the claim
    result = process_json(input_data)

    assert "url_2" not in result  # No perturbations should alter the claim


def test_process_json_no_perturbations_alter():
    """Test process_json when no perturbations alter the claim.

    Verifies that the URL is excluded from the output.
    """
    input_data = {
        "url_3": {
            "claim": "Static claim without numbers",
            "entities": [],  # No entities
        }
    }
    result = process_json(input_data)
    assert "url_3" not in result


def test_process_json_mixed_perturbations():
    """Test process_json with mixed perturbation outcomes.

    Some perturbations alter the claim, while others do not.
    """
    input_data = {
        "url_4": {
            "claim": "Population is 1000 in city",
            "entities": [
                {"start_char": 14, "end_char": 18, "label": "CARDINAL"}
            ],
        }
    }
    # Mock perturbation functions: random_replacement alters,
    # masking does not alter, value_removal alters
    with (
        patch(
            "code.data_preprocessing.perturbations.obfuscate.random_replacement",  # noqa: E501
            return_value="Population is 2000 in city",
        ),
        patch(
            "code.data_preprocessing.perturbations.obfuscate.masking",
            return_value="Population is 1000 in city",
        ),
        patch(
            "code.data_preprocessing.perturbations.obfuscate.value_removal",
            return_value="Population is in city",
        ),
    ):

        result = process_json(input_data)

        assert "url_4" in result
        assert "random_replacement_false" in result["url_4"]
        assert "masking_false" not in result["url_4"]  # No alteration
        assert "value_removal_false" in result["url_4"]
        assert (
            result["url_4"]["random_replacement_false"]
            == "Population is 2000 in city"
        )
        assert (
            result["url_4"]["value_removal_false"] == "Population is in city"
        )  # noqa: E501


def test_process_json_multiple_urls():
    """Test process_json with multiple URLs, some altered and some not."""
    input_data = {
        "url_5": {
            "claim": "GDP was 5000 dollars",
            "entities": [{"start_char": 8, "end_char": 12, "label": "MONEY"}],
        },
        "url_6": {
            "claim": "No numerical data here",
            "entities": [],
        },
        "url_7": {
            "claim": "Growth rate is 5%",
            "entities": [
                {"start_char": 15, "end_char": 17, "label": "PERCENT"}
            ],
        },
    }
    # Mock perturbation functions:
    # For url_5: all perturbations alter
    # For url_6: no perturbations alter
    # For url_7: only masking alters
    with (
        patch(
            "code.data_preprocessing.perturbations.obfuscate.random_replacement"
        ) as mock_random_replacement,
        patch(
            "code.data_preprocessing.perturbations.obfuscate.masking"
        ) as mock_masking,
        patch(
            "code.data_preprocessing.perturbations.obfuscate.value_removal"
        ) as mock_value_removal,
    ):

        # Define side effects based on input URL
        def side_effect_random_replacement(claim, entities):
            if "GDP was 5000 dollars" in claim:
                return "GDP was 6000 dollars"
            elif "Growth rate is 5%" in claim:
                return "Growth rate is 5%"  # No change for random_replacement
            else:
                return claim  # Return original claim for unmatched cases

        def side_effect_masking(claim, entities):
            if "GDP was 5000 dollars" in claim:
                return "GDP was $#### dollars"
            elif "Growth rate is 5%" in claim:
                return "Growth rate is ##%"
            else:
                return claim  # Return original claim for unmatched cases

        def side_effect_value_removal(claim, entities):
            if "GDP was 5000 dollars" in claim:
                return "GDP was dollars"
            elif "Growth rate is 5%" in claim:
                return "Growth rate is 5%"  # No change for value_removal
            else:
                return claim  # Return original claim for unmatched cases

        mock_random_replacement.side_effect = side_effect_random_replacement
        mock_masking.side_effect = side_effect_masking
        mock_value_removal.side_effect = side_effect_value_removal

        result = process_json(input_data)

        # url_5 should have all perturbations
        assert "url_5" in result
        assert (
            result["url_5"]["random_replacement_false"]
            == "GDP was 6000 dollars"
        )
        assert result["url_5"]["masking_false"] == "GDP was $#### dollars"
        assert result["url_5"]["value_removal_false"] == "GDP was dollars"

        # url_6 should not be in the result
        assert "url_6" not in result

        # url_7 should only have masking altered
        assert "url_7" in result
        assert "random_replacement_false" not in result["url_7"]
        assert "masking_false" in result["url_7"]
        assert "value_removal_false" not in result["url_7"]
        assert result["url_7"]["masking_false"] == "Growth rate is ##%"


def test_process_json_no_entities():
    """Test process_json when claims have no entities.

    Ensures that claims without entities are excluded from the output.
    """
    input_data = {
        "url_8": {
            "claim": "Just a plain statement",
            "entities": [],
        }
    }
    result = process_json(input_data)
    assert "url_8" not in result


def test_process_json_non_numerical_entities():
    """Test process_json when entities are non-numerical.

    Ensures that non-numerical entities do not trigger perturbations.
    """
    input_data = {
        "url_9": {
            "claim": "The quick brown fox",
            "entities": [{"start_char": 4, "end_char": 9, "label": "ANIMAL"}],
        }
    }
    result = process_json(input_data)
    assert "url_9" not in result


def test_process_json_overlapping_entities():
    """Test process_json with overlapping entities.

    Ensures that overlapping numerical entities are handled correctly.
    """
    input_data = {
        "url_10": {
            "claim": "The event starts at 10:00 AM on 2023-09-15",
            "entities": [
                {"start_char": 19, "end_char": 24, "label": "TIME"},  # "10:00"
                {
                    "start_char": 28,
                    "end_char": 38,
                    "label": "DATE",
                },  # "2023-09-15"
            ],
        }
    }
    # Mock perturbation functions to alter both entities
    with (
        patch(
            "code.data_preprocessing.perturbations.obfuscate.random_replacement"
        ) as mock_random_replacement,
        patch(
            "code.data_preprocessing.perturbations.obfuscate.masking"
        ) as mock_masking,
        patch(
            "code.data_preprocessing.perturbations.obfuscate.value_removal"
        ) as mock_value_removal,
    ):

        mock_random_replacement.side_effect = (
            lambda claim, entities: claim.replace("10:00", "11:00").replace(
                "2023-09-15", "2024-10-16"
            )
        )
        mock_masking.side_effect = lambda claim, entities: claim.replace(
            "10:00", "##:##"
        ).replace("2023-09-15", "####-##-##")
        mock_value_removal.side_effect = lambda claim, entities: claim.replace(
            "10:00 ", ""
        ).replace("2023-09-15", "")

        result = process_json(input_data)

        assert "url_10" in result
        assert (
            result["url_10"]["random_replacement_false"]
            == "The event starts at 11:00 AM on 2024-10-16"
        )
        assert (
            result["url_10"]["masking_false"]
            == "The event starts at ##:## AM on ####-##-##"
        )
        assert (
            result["url_10"]["value_removal_false"]
            == "The event starts at AM on "
        )
