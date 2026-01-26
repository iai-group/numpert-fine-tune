"""Test cases for numeration perturbations in data preprocessing."""

from unittest.mock import patch

from code.data_preprocessing.perturbations.numeration import (
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

    @patch("code.data_preprocessing.perturbations.numeration.process_entity")
    def test_process_sentence_multiple_entities_sorted(
        self, mock_process_entity
    ):
        """Test process_sentence processes entities in reverse order."""
        mock_process_entity.side_effect = [
            ("First change", True),
            ("Second change", True),
        ]

        # Entities should be processed in reverse order
        calls = mock_process_entity.call_args_list
        assert (
            calls[0][0][0]["start_char"] == 10
        )  # First call with higher start_char
        assert (
            calls[1][0][0]["start_char"] == 5
        )  # Second call with lower start_char


class TestProcessJson:
    """Test process_json function to apply numeration perturbations."""

    @patch(
        "code.data_preprocessing.perturbations.utils.load_save.INCLUDED_KEYS",
        ["id", "claim"],
    )
    @patch("code.data_preprocessing.perturbations.numeration.process_sentence")
    def test_process_json_no_changes(self, mock_process_sentence):
        """Test process_json with no changes."""
        mock_process_sentence.return_value = None
        input_data = [{"id": 1, "claim": "test"}]
        result = process_json(input_data)
        assert result == []

    @patch(
        "code.data_preprocessing.perturbations.utils.load_save.INCLUDED_KEYS",
        ["id", "claim"],
    )
    @patch("code.data_preprocessing.perturbations.numeration.process_sentence")
    def test_process_json_with_changes(self, mock_process_sentence):
        """Test process_json with changes."""
        mock_process_sentence.side_effect = ["true result", "false result"]
        input_data = [{"id": 1, "claim": "test", "extra": "removed"}]
        result = process_json(input_data)

        expected = [
            {
                "id": 1,
                "claim": "test",
                "numeration_true": "true result",
                "numeration_false": "false result",
            }
        ]
        assert result == expected

    @patch(
        "code.data_preprocessing.perturbations.utils.load_save.INCLUDED_KEYS",
        ["id"],
    )
    @patch("code.data_preprocessing.perturbations.numeration.process_sentence")
    def test_process_json_only_false_change(self, mock_process_sentence):
        """Test process_json with only false change."""
        mock_process_sentence.side_effect = [None, "false result"]
        input_data = [{"id": 1}]
        result = process_json(input_data)

        expected = [{"id": 1, "numeration_false": "false result"}]
        assert result == expected


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
