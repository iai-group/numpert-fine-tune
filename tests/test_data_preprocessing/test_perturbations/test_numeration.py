"""Test cases for numeration perturbations in data preprocessing."""

from unittest.mock import patch

from src.data_preprocessing.perturbations.numeration import (
    modify_number,
    numeration_perturbation,
    adjust_entity_bounds,
    apply_text_replacement,
    clean_sentence,
    process_entity,
    process_sentence,
    process_json,
    numerical_entities,
    perturbation_factors,
)


class TestModifyNumber:
    """Test modify_number function to perturb numerical values."""

    def test_modify_number_true_returns_original(self):
        """Test case for true value, should return original number."""
        assert modify_number(50.0, "PERCENT", "true") == 50.0
        assert modify_number(100.0, "MONEY", "true") == 100.0

    def test_modify_number_percent_false(self):
        """Test case for PERCENT entity type."""
        result = modify_number(50.0, "PERCENT", "false")
        assert result == 55.0  # 50 * 1.1 = 55

    def test_modify_number_money_false(self):
        """Test case for MONEY entity type."""
        result = modify_number(100.0, "MONEY", "false")
        assert result == 110.0  # 100 * 1.1 = 110

    def test_modify_number_cardinal_false(self):
        """Test case for CARDINAL entity type."""
        result = modify_number(10.0, "CARDINAL", "false")
        assert result == 11.0  # 10 * 1.1 = 11

    def test_modify_number_date_false(self):
        """Test case for DATE entity type."""
        result = modify_number(2020.0, "DATE", "false")
        assert result == 2025.0  # 2020 + 5

    def test_modify_number_time_false(self):
        """Test case for TIME entity type."""
        result = modify_number(120.0, "TIME", "false")
        assert result == 180.0  # 120 + 60

    def test_modify_number_unknown_entity_false(self):
        """Test case for unknown entity type, should add default factor."""
        result = modify_number(50.0, "UNKNOWN", "false")
        assert result == 51.0  # 50 + 1 (default factor)

    def test_modify_number_rounding_adjustment(self):
        """Test case where rounding produces same integer, should add 1."""
        # Test case where rounding produces same integer
        result = modify_number(9.0, "PERCENT", "false")
        assert result == 10.0  # 9 * 1.1 = 9.9, rounds to 10

    def test_modify_number_small_number_adjustment(self):
        """Test case where small number rounds to same value, should add 1."""
        # Test case where modification rounds to same value, should add 1
        result = modify_number(1.0, "PERCENT", "false")
        assert result == 2.0  # 1 * 1.1 = 1.1, rounds to 1, then +1 = 2


class TestNumerationPerturbation:
    """Test numeration_perturbation function."""

    def test_numeration_perturbation_no_digits(self):
        """Test numeration_perturbation with no digits in the text."""
        result = numeration_perturbation("no digits here", "PERCENT", "false")
        assert result == "no digits here"

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_percent(self, mock_num2words):
        """Test numeration_perturbation with a percent value."""
        mock_num2words.return_value = "fifty-five"
        result = numeration_perturbation("50%", "PERCENT", "false")
        assert result == "fifty-five percent"

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_money(self, mock_num2words):
        """Test numeration_perturbation with a money value."""
        mock_num2words.return_value = "one hundred ten"
        result = numeration_perturbation("$100", "MONEY", "false")
        assert result == "one hundred ten dollars"

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_cardinal_with_unit(self, mock_num2words):
        """Test numeration_perturbation with a cardinal number and unit."""
        mock_num2words.return_value = "eleven"
        result = numeration_perturbation("10 people", "CARDINAL", "false")
        assert result == "eleven people"

    def test_numeration_perturbation_true_value(self):
        """Test numeration_perturbation returns original value for 'true'."""
        result = numeration_perturbation("50%", "PERCENT", "true")
        assert result == "fifty percent"

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_with_commas(self, mock_num2words):
        """Test numeration_perturbation with numbers containing commas."""
        mock_num2words.return_value = "one thousand one hundred"
        result = numeration_perturbation("1,000", "CARDINAL", "false")
        assert result == "one thousand one hundred"

    def test_numeration_perturbation_exception_handling(self):
        """Test numeration_perturbation handles exceptions gracefully."""
        with patch(
            "code.data_preprocessing.perturbations.numeration.num2words",
            side_effect=Exception("test error"),
        ):
            with patch("builtins.print") as mock_print:
                result = numeration_perturbation("50%", "PERCENT", "false")
                assert result == "50%"
                mock_print.assert_called()


class TestAdjustEntityBounds:
    """Test adjust_entity_bounds function to handle numerical entities."""

    def test_adjust_entity_bounds_money(self):
        """Test adjust_entity_bounds adjusts bounds for money entities."""
        sentence = "I have $50 here"
        entity = {"label": "MONEY"}
        start, end = adjust_entity_bounds(sentence, entity, 7, 9)
        assert start == 7
        assert end == 9

    def test_adjust_entity_bounds_money_with_symbol(self):
        """Test adjust_entity_bounds adjusts bounds for money entities."""
        sentence = "I have $50 here"
        entity = {"label": "MONEY"}
        start, end = adjust_entity_bounds(sentence, entity, 8, 10)
        assert start == 7  # Adjusted to include $
        assert end == 10

    def test_adjust_entity_bounds_percent(self):
        """Test adjust_entity_bounds adjusts bounds for percent entities."""
        sentence = "50% is good"
        entity = {"label": "PERCENT"}
        start, end = adjust_entity_bounds(sentence, entity, 0, 2)
        assert start == 0
        assert end == 3  # Adjusted to include %

    def test_adjust_entity_bounds_other_entity(self):
        """Test to not change bounds for non-money entities."""
        sentence = "The year 2020"
        entity = {"label": "DATE"}
        start, end = adjust_entity_bounds(sentence, entity, 9, 13)
        assert start == 9
        assert end == 13


class TestApplyTextReplacement:
    """Test apply_text_replacement function."""

    def test_apply_text_replacement(self):
        """Test apply_text_replacement replaces text correctly."""
        result = apply_text_replacement("Hello world!", "beautiful", 6, 11)
        assert result == "Hello beautiful!"


class TestCleanSentence:
    """Test clean_sentence function."""

    def test_clean_sentence_multiple_spaces(self):
        """Test clean_sentence removes multiple spaces."""
        result = clean_sentence("Hello    world   !")
        assert result == "Hello world!"

    def test_clean_sentence_punctuation_spacing(self):
        """Test clean_sentence removes extra spaces around punctuation."""
        result = clean_sentence("Hello , world !")
        assert result == "Hello, world!"

    def test_clean_sentence_mixed_issues(self):
        """Test clean_sentence handles mixed spacing and punctuation issues."""
        result = clean_sentence("  Hello  ,  world  !  ")
        assert result == "Hello, world!"


class TestProcessEntity:
    """Test process_entity function to perturb numerical entities."""

    def test_process_entity_non_numerical(self):
        """Test process_entity does not change non-numerical entities."""
        entity = {"label": "PERSON", "text": "John"}
        sentence = "John is here"
        result_sentence, changed = process_entity(entity, "false", sentence)
        assert result_sentence == "John is here"
        assert changed is False

    @patch(
        "code.data_preprocessing.perturbations.numeration.numeration_perturbation"  # noqa: E501
    )
    def test_process_entity_no_change(self, mock_perturbation):
        """Test process_entity does not change numerical entities."""
        mock_perturbation.return_value = "50%"
        entity = {
            "label": "PERCENT",
            "text": "50%",
            "start_char": 0,
            "end_char": 3,
        }
        sentence = "50% is good"
        result_sentence, changed = process_entity(entity, "false", sentence)
        assert result_sentence == "50% is good"
        assert changed is False

    @patch(
        "code.data_preprocessing.perturbations.numeration.numeration_perturbation"  # noqa: E501
    )
    def test_process_entity_with_change(self, mock_perturbation):
        """Test process_entity modifies numerical entities."""
        mock_perturbation.return_value = "sixty percent"
        entity = {
            "label": "PERCENT",
            "text": "50%",
            "start_char": 0,
            "end_char": 3,
        }
        sentence = "50% is good"
        result_sentence, changed = process_entity(entity, "false", sentence)
        assert result_sentence == "sixty percent is good"
        assert changed is True


class TestProcessSentence:
    """Test process_sentence function to perturb numerical entities."""

    def test_process_sentence_no_entities(self):
        """Test process_sentence returns None if no entities are present."""
        content = {"normalized_claim": "Hello world", "entities": []}
        result = process_sentence(content, "false")
        assert result is None

    @patch("code.data_preprocessing.perturbations.numeration.process_entity")
    def test_process_sentence_with_changes(self, mock_process_entity):
        """Test process_sentence applies changes to entities."""
        mock_process_entity.return_value = ("Modified sentence", True)
        content = {
            "normalized_claim": "Original sentence",
            "entities": [{"label": "PERCENT", "start_char": 0}],
        }
        result = process_sentence(content, "false")
        assert result == "Modified sentence"


class TestConstants:
    """Test constants related to numerical entities and perturbation factors."""

    def test_numerical_entities_contains_expected_types(self):
        """Test numerical_entities contains expected types."""
        expected = {"CARDINAL", "DATE", "PERCENT", "MONEY", "QUANTITY", "TIME"}
        assert numerical_entities == expected

    def test_perturbation_factors_contains_expected_values(self):
        """Test perturbation_factors contains expected values."""
        expected = {
            "PERCENT": 0.1,
            "MONEY": 0.1,
            "CARDINAL": 0.1,
            "DATE": 5,
            "TIME": 60,
        }
        assert perturbation_factors == expected
class TestProcessJsonWithLabel:
    """Test process_json function with label-based key naming."""

    @patch(
        "code.data_preprocessing.perturbations.numeration.process_sentence"
    )
    def test_process_json_label_used_in_key(self, mock_process_sentence):
        """Test process_json uses label in output key names."""
        mock_process_sentence.side_effect = ["true result", "false result"]
        input_data = [{"id": 1, "label": "True"}]
        with patch(
            "code.data_preprocessing.perturbations.numeration.INCLUDED_KEYS",
            ["id", "label"],
        ):
            result = process_json(input_data)
        assert "perturb_type1_True" in result[0]
        assert "perturb_type2_True" in result[0]

    @patch(
        "code.data_preprocessing.perturbations.numeration.process_sentence"
    )
    def test_process_json_unknown_label_fallback(self, mock_process_sentence):
        """Test process_json falls back to UNKNOWN when label is missing."""
        mock_process_sentence.side_effect = [None, "false result"]
        input_data = [{"id": 1}]
        with patch(
            "code.data_preprocessing.perturbations.numeration.INCLUDED_KEYS",
            ["id"],
        ):
            result = process_json(input_data)
        assert "perturb_type2_UNKNOWN" in result[0]

    @patch(
        "code.data_preprocessing.perturbations.numeration.process_sentence"
    )
    def test_process_json_multiple_entries(self, mock_process_sentence):
        """Test process_json processes multiple entries correctly."""
        mock_process_sentence.side_effect = [
            "true result 1",
            "false result 1",
            None,
            None,
            "true result 3",
            None,
        ]
        input_data = [
            {"id": 1, "label": "True"},
            {"id": 2, "label": "False"},
            {"id": 3, "label": "True"},
        ]
        with patch(
            "code.data_preprocessing.perturbations.numeration.INCLUDED_KEYS",
            ["id", "label"],
        ):
            result = process_json(input_data)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 3

    @patch(
        "code.data_preprocessing.perturbations.numeration.process_sentence"
    )
    def test_process_json_excluded_keys_not_present(
        self, mock_process_sentence
    ):
        """Test process_json excludes keys not in INCLUDED_KEYS."""
        mock_process_sentence.side_effect = ["true result", None]
        input_data = [{"id": 1, "label": "True", "extra_field": "should_be_excluded"}]
        with patch(
            "code.data_preprocessing.perturbations.numeration.INCLUDED_KEYS",
            ["id", "label"],
        ):
            result = process_json(input_data)
        assert "extra_field" not in result[0]


class TestModifyNumberEdgeCases:
    """Additional edge case tests for modify_number."""

    def test_modify_number_zero_percent(self):
        """Test modify_number with zero value for PERCENT."""
        result = modify_number(0.0, "PERCENT", "false")
        # 0 * 1.1 = 0.0, rounds to 0, same as original, so +1 = 1
        assert result == 1

    def test_modify_number_large_number(self):
        """Test modify_number with large numbers."""
        result = modify_number(1000.0, "CARDINAL", "false")
        assert result == 1100  # 1000 * 1.1 = 1100

    def test_modify_number_quantity_entity(self):
        """Test modify_number with QUANTITY entity type (uses default factor)."""
        result = modify_number(50.0, "QUANTITY", "false")
        assert result == 51.0  # 50 + 1 (default factor)

    def test_modify_number_true_preserves_float(self):
        """Test that true value preserves float precision."""
        result = modify_number(3.14, "CARDINAL", "true")
        assert result == 3.14


class TestNumerationPerturbationEdgeCases:
    """Additional edge case tests for numeration_perturbation."""

    def test_numeration_perturbation_default_truth_value(self):
        """Test numeration_perturbation uses 'true' as default."""
        result = numeration_perturbation("50%", "PERCENT")
        # Default is 'true', so no modification but converts to words
        assert "fifty" in result

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_date(self, mock_num2words):
        """Test numeration_perturbation with a date value."""
        mock_num2words.return_value = "two thousand twenty-five"
        result = numeration_perturbation("2020", "DATE", "false")
        assert "two thousand twenty-five" in result

    @patch("code.data_preprocessing.perturbations.numeration.num2words")
    def test_numeration_perturbation_money_singular(self, mock_num2words):
        """Test numeration_perturbation with $1 (singular dollar)."""
        mock_num2words.return_value = "one"
        result = numeration_perturbation("$1", "MONEY", "true")
        assert result == "one dollar"

    def test_numeration_perturbation_empty_string(self):
        """Test numeration_perturbation with empty string."""
        result = numeration_perturbation("", "PERCENT", "false")
        assert result == ""


class TestProcessSentenceEdgeCases:
    """Additional edge case tests for process_sentence."""

    def test_process_sentence_no_normalized_claim_key(self):
        """Test process_sentence with missing normalized_claim raises KeyError."""
        content = {"entities": []}
        try:
            process_sentence(content, "false")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    @patch("code.data_preprocessing.perturbations.numeration.process_entity")
    def test_process_sentence_no_perturbation_made(self, mock_process_entity):
        """Test process_sentence returns None when no perturbation is made."""
        mock_process_entity.return_value = ("Original sentence", False)
        content = {
            "normalized_claim": "Original sentence",
            "entities": [{"label": "PERCENT", "start_char": 0}],
        }
        result = process_sentence(content, "false")
        assert result is None

    @patch("code.data_preprocessing.perturbations.numeration.process_entity")
    def test_process_sentence_entities_sorted_reverse(
        self, mock_process_entity
    ):
        """Test process_sentence sorts entities in reverse start_char order."""
        mock_process_entity.return_value = ("sentence", False)
        content = {
            "normalized_claim": "50% and 20 people",
            "entities": [
                {"label": "PERCENT", "start_char": 5},
                {"label": "CARDINAL", "start_char": 10},
                {"label": "DATE", "start_char": 2},
            ],
        }
        process_sentence(content, "false")
        calls = mock_process_entity.call_args_list
        start_chars = [call[0][0]["start_char"] for call in calls]
        assert start_chars == sorted(start_chars, reverse=True)


class TestAdjustEntityBoundsEdgeCases:
    """Additional edge case tests for adjust_entity_bounds."""

    def test_adjust_entity_bounds_start_at_zero_money(self):
        """Test adjust_entity_bounds does not go below 0 for MONEY."""
        sentence = "$50 here"
        entity = {"label": "MONEY"}
        start, end = adjust_entity_bounds(sentence, entity, 0, 3)
        assert start == 0  # Cannot go below 0

    def test_adjust_entity_bounds_percent_at_end(self):
        """Test adjust_entity_bounds handles percent at end of sentence."""
        sentence = "50%"
        entity = {"label": "PERCENT"}
        start, end = adjust_entity_bounds(sentence, entity, 0, 2)
        assert end == 3  # Should include the %

    def test_adjust_entity_bounds_percent_no_symbol(self):
        """Test adjust_entity_bounds when no % symbol follows."""
        sentence = "50 percent"
        entity = {"label": "PERCENT"}
        start, end = adjust_entity_bounds(sentence, entity, 0, 2)
        assert end == 2  # No % to include

    def test_adjust_entity_bounds_euro_symbol(self):
        """Test adjust_entity_bounds handles € symbol for MONEY."""
        sentence = "€100 total"
        entity = {"label": "MONEY"}
        start, end = adjust_entity_bounds(sentence, entity, 1, 4)
        assert start == 0  # Adjusted to include €

