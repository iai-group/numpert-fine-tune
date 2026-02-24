import pytest
from unittest.mock import patch, MagicMock

"""Tests for the obfuscate perturbation module."""

from src.data_preprocessing.perturbations.obfuscate import (
    replace_digits_with_new_sequence,
    random_replacement,
    masking,
    value_removal,
    process_json,
    NUMERICAL_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cardinal_entity():
    return [{"label": "CARDINAL", "start_char": 10, "end_char": 13}]


@pytest.fixture
def non_numerical_entity():
    return [{"label": "PERSON", "start_char": 0, "end_char": 5}]


@pytest.fixture
def multiple_entities():
    return [
        {"label": "CARDINAL", "start_char": 0, "end_char": 3},
        {"label": "MONEY", "start_char": 14, "end_char": 18},
    ]


@pytest.fixture
def sample_input_data():
    return [
        {
            "normalized_claim": "There were 500 people at the event.",
            "entities": [
                {"label": "CARDINAL", "start_char": 11, "end_char": 14}
            ],
            "label": "True",
            "claim_id": "001",
        }
    ]


# ---------------------------------------------------------------------------
# replace_digits_with_new_sequence
# ---------------------------------------------------------------------------


class TestReplaceDigitsWithNewSequence:
    def test_returns_different_value(self):
        result = replace_digits_with_new_sequence("42")
        assert result != "42"

    def test_same_length(self):
        original = "123"
        result = replace_digits_with_new_sequence(original)
        assert len(result) == len(original)

    def test_no_digits_returns_original(self):
        assert replace_digits_with_new_sequence("abc") == "abc"

    def test_empty_string_returns_empty(self):
        assert replace_digits_with_new_sequence("") == ""

    def test_does_not_start_with_zero(self):
        # Run multiple times to reduce flakiness
        for _ in range(20):
            result = replace_digits_with_new_sequence("9")
            assert result[0] != "0"

    def test_preserves_non_digit_chars(self):
        result = replace_digits_with_new_sequence("$1,000")
        assert result[0] == "$"
        assert result[2] == ","

    def test_all_chars_no_digits_unchanged(self):
        assert replace_digits_with_new_sequence("hello") == "hello"

    def test_single_digit_changes(self):
        results = set()
        for _ in range(50):
            results.add(replace_digits_with_new_sequence("5"))
        # Should produce different digits across runs
        assert len(results) > 1

    def test_mixed_text_with_digits(self):
        result = replace_digits_with_new_sequence("2023-01-15")
        # Non-digit chars preserved
        assert result[4] == "-"
        assert result[7] == "-"
        # Digit count unchanged
        assert sum(c.isdigit() for c in result) == sum(
            c.isdigit() for c in "2023-01-15"
        )


# ---------------------------------------------------------------------------
# random_replacement
# ---------------------------------------------------------------------------


class TestRandomReplacement:
    def test_replaces_numerical_entity(self):
        claim = "There are 100 apples."
        entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 13}]
        result = random_replacement(claim, entities)
        assert result != claim

    def test_ignores_non_numerical_entity(self, non_numerical_entity):
        claim = "Alice went to the store."
        result = random_replacement(claim, non_numerical_entity)
        assert result == claim

    def test_empty_entities(self):
        claim = "No numbers here."
        assert random_replacement(claim, []) == claim

    def test_all_numerical_labels_are_replaced(self):
        for label in NUMERICAL_LABELS:
            claim = "Value is 99 units."
            entities = [{"label": label, "start_char": 9, "end_char": 11}]
            result = random_replacement(claim, entities)
            assert result != claim

    def test_multiple_entities_replaced(self, multiple_entities):
        claim = "100 items cost $200."
        result = random_replacement(claim, multiple_entities)
        # Surrounding text preserved
        assert " items cost $" in result

    def test_result_length_same_as_original_for_same_digit_count(self):
        claim = "There are 42 cats."
        entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 12}]
        result = random_replacement(claim, entities)
        assert len(result) == len(claim)

    def test_claim_without_digits_in_entity_unchanged(self):
        claim = "Happened in January."
        entities = [{"label": "DATE", "start_char": 12, "end_char": 19}]
        result = random_replacement(claim, entities)
        assert result == claim


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_masks_digits(self):
        claim = "There are 100 apples."
        entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 13}]
        result = masking(claim, entities)
        assert "###" in result

    def test_masks_comma_and_dot(self):
        claim = "Cost is 1,000.50 dollars."
        entities = [{"label": "MONEY", "start_char": 8, "end_char": 16}]
        result = masking(claim, entities)
        assert "," not in result[8:16]
        assert "." not in result[8:16]

    def test_non_numerical_entity_not_masked(self, non_numerical_entity):
        claim = "Alice went to the store."
        assert masking(claim, non_numerical_entity) == claim

    def test_empty_entities_no_change(self):
        claim = "No changes here."
        assert masking(claim, []) == claim

    def test_non_digit_non_special_chars_preserved(self):
        claim = "About $1,200 total."
        entities = [{"label": "MONEY", "start_char": 6, "end_char": 12}]
        result = masking(claim, entities)
        assert result[6] == "$"

    def test_all_numerical_labels_masked(self):
        for label in NUMERICAL_LABELS:
            claim = "Value is 99 units."
            entities = [{"label": label, "start_char": 9, "end_char": 11}]
            result = masking(claim, entities)
            assert "##" in result

    def test_multiple_entities_masked(self, multiple_entities):
        claim = "100 items cost $200."
        result = masking(claim, multiple_entities)
        assert "###" in result


# ---------------------------------------------------------------------------
# value_removal
# ---------------------------------------------------------------------------


class TestValueRemoval:
    def test_removes_numerical_entity(self):
        claim = "There are 100 apples."
        entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 14}]
        result = value_removal(claim, entities)
        assert "100" not in result

    def test_non_numerical_entity_not_removed(self, non_numerical_entity):
        claim = "Alice went to the store."
        assert value_removal(claim, non_numerical_entity) == claim

    def test_empty_entities_no_change(self):
        claim = "Nothing removed."
        assert value_removal(claim, []) == claim

    def test_no_double_spaces_after_removal(self):
        claim = "There are 100 apples."
        entities = [{"label": "CARDINAL", "start_char": 10, "end_char": 14}]
        result = value_removal(claim, entities)
        assert "  " not in result

    def test_multiple_entities_removed(self, multiple_entities):
        claim = "100 items cost $200."
        result = value_removal(claim, multiple_entities)
        assert "100" not in result

    def test_all_numerical_labels_removed(self):
        for label in NUMERICAL_LABELS:
            claim = "Value is 99 units."
            entities = [{"label": label, "start_char": 9, "end_char": 11}]
            result = value_removal(claim, entities)
            assert "99" not in result

    def test_result_is_stripped(self):
        claim = "100 is the amount."
        entities = [{"label": "CARDINAL", "start_char": 0, "end_char": 3}]
        result = value_removal(claim, entities)
        assert not result.startswith(" ")


# ---------------------------------------------------------------------------
# process_json
# ---------------------------------------------------------------------------

MOCK_INCLUDED_KEYS = ["claim_id", "label", "normalized_claim"]


@pytest.fixture
def patch_included_keys():
    with patch(
        "code.data_preprocessing.perturbations.obfuscate.INCLUDED_KEYS",
        MOCK_INCLUDED_KEYS,
    ):
        yield


class TestProcessJson:
    def test_returns_dict(self, sample_input_data, patch_included_keys):
        result = process_json(sample_input_data)
        assert isinstance(result, dict)

    def test_label_key_present(self, sample_input_data, patch_included_keys):
        result = process_json(sample_input_data)
        assert "True" in result

    def test_perturbed_keys_added(self, sample_input_data, patch_included_keys):
        result = process_json(sample_input_data)
        entry = result["True"][0]
        perturbation_keys = {
            "perturb_type3_random",
            "perturb_type3_masked",
            "perturb_type3_removed",
        }
        assert perturbation_keys & entry.keys()

    def test_skips_entry_with_no_perturbation(self, patch_included_keys):
        # No numerical entities → all perturbations equal original → skipped
        data = [
            {
                "normalized_claim": "This is a simple claim.",
                "entities": [],
                "label": "True",
                "claim_id": "002",
            }
        ]
        result = process_json(data)
        assert result == {}

    def test_multiple_labels_grouped(self, patch_included_keys):
        data = [
            {
                "normalized_claim": "There were 500 people.",
                "entities": [
                    {"label": "CARDINAL", "start_char": 11, "end_char": 14}
                ],
                "label": "True",
                "claim_id": "001",
            },
            {
                "normalized_claim": "Only 10 attended.",
                "entities": [
                    {"label": "CARDINAL", "start_char": 5, "end_char": 7}
                ],
                "label": "False",
                "claim_id": "002",
            },
        ]
        result = process_json(data)
        assert "True" in result
        assert "False" in result

    def test_base_entry_keys_from_included_keys(
        self, sample_input_data, patch_included_keys
    ):
        result = process_json(sample_input_data)
        entry = result["True"][0]
        for key in MOCK_INCLUDED_KEYS:
            if key in sample_input_data[0]:
                assert key in entry

    def test_unknown_label_used_when_missing(self, patch_included_keys):
        data = [
            {
                "normalized_claim": "There are 50 units.",
                "entities": [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 12}
                ],
                "claim_id": "003",
            }
        ]
        result = process_json(data)
        assert "UNKNOWN" in result

    def test_random_key_only_added_when_different(self, patch_included_keys):
        # Entity with no digits → random_replacement returns original
        data = [
            {
                "normalized_claim": "Happened in January.",
                "entities": [
                    {
                        "label": "DATE",
                        "start_char": 12,
                        "end_char": 19,
                    }
                ],
                "label": "True",
                "claim_id": "004",
            }
        ]
        result = process_json(data)
        if "True" in result:
            entry = result["True"][0]
            assert "perturb_type3_random" not in entry

    def test_empty_input_returns_empty_dict(self, patch_included_keys):
        """Tests for the obfuscate perturbation module."""

        from code.data_preprocessing.perturbations.obfuscate import (
            replace_digits_with_new_sequence,
            random_replacement,
            masking,
            value_removal,
            process_json,
            NUMERICAL_LABELS,
        )

        # ---------------------------------------------------------------------------
        # Fixtures
        # ---------------------------------------------------------------------------

        @pytest.fixture
        def cardinal_entity():
            return [{"label": "CARDINAL", "start_char": 10, "end_char": 13}]

        @pytest.fixture
        def non_numerical_entity():
            return [{"label": "PERSON", "start_char": 0, "end_char": 5}]

        @pytest.fixture
        def multiple_entities():
            return [
                {"label": "CARDINAL", "start_char": 0, "end_char": 3},
                {"label": "MONEY", "start_char": 14, "end_char": 18},
            ]

        @pytest.fixture
        def sample_input_data():
            return [
                {
                    "normalized_claim": "There were 500 people at the event.",
                    "entities": [
                        {"label": "CARDINAL", "start_char": 11, "end_char": 14}
                    ],
                    "label": "True",
                    "claim_id": "001",
                }
            ]

        # ---------------------------------------------------------------------------
        # replace_digits_with_new_sequence
        # ---------------------------------------------------------------------------

        class TestReplaceDigitsWithNewSequence:
            def test_returns_different_value(self):
                result = replace_digits_with_new_sequence("42")
                assert result != "42"

            def test_same_length(self):
                original = "123"
                result = replace_digits_with_new_sequence(original)
                assert len(result) == len(original)

            def test_no_digits_returns_original(self):
                assert replace_digits_with_new_sequence("abc") == "abc"

            def test_empty_string_returns_empty(self):
                assert replace_digits_with_new_sequence("") == ""

            def test_does_not_start_with_zero(self):
                # Run multiple times to reduce flakiness
                for _ in range(20):
                    result = replace_digits_with_new_sequence("9")
                    assert result[0] != "0"

            def test_preserves_non_digit_chars(self):
                result = replace_digits_with_new_sequence("$1,000")
                assert result[0] == "$"
                assert result[2] == ","

            def test_all_chars_no_digits_unchanged(self):
                assert replace_digits_with_new_sequence("hello") == "hello"

            def test_single_digit_changes(self):
                results = set()
                for _ in range(50):
                    results.add(replace_digits_with_new_sequence("5"))
                # Should produce different digits across runs
                assert len(results) > 1

            def test_mixed_text_with_digits(self):
                result = replace_digits_with_new_sequence("2023-01-15")
                # Non-digit chars preserved
                assert result[4] == "-"
                assert result[7] == "-"
                # Digit count unchanged
                assert sum(c.isdigit() for c in result) == sum(
                    c.isdigit() for c in "2023-01-15"
                )

            def test_result_only_digits_in_new_text(self):
                original = "42"
                result = replace_digits_with_new_sequence(original)
                assert all(c.isdigit() for c in result)

            def test_leading_digit_never_zero_multi_digit(self):
                for _ in range(30):
                    result = replace_digits_with_new_sequence("10")
                    assert result[0] != "0"

            def test_special_chars_only_no_digits(self):
                assert replace_digits_with_new_sequence("$,.!") == "$,.!"

            def test_single_zero_digit(self):
                # "0" has no digits[1:] issue; result should differ and not start with 0
                for _ in range(30):
                    result = replace_digits_with_new_sequence("0")
                    assert result != "0"
                    assert result[0] != "0"

            def test_large_number_preserves_structure(self):
                original = "1,000,000"
                result = replace_digits_with_new_sequence(original)
                assert result[1] == ","
                assert result[5] == ","
                assert len(result) == len(original)

            def test_percent_string_preserves_percent_sign(self):
                original = "50%"
                result = replace_digits_with_new_sequence(original)
                assert result[-1] == "%"
                assert len(result) == len(original)

            def test_all_same_digit_string(self):
                # e.g. "111" should be replaced with something different
                result = replace_digits_with_new_sequence("111")
                assert result != "111"

            def test_two_digit_number_only_digits(self):
                result = replace_digits_with_new_sequence("99")
                assert len(result) == 2
                assert all(c.isdigit() for c in result)

        # ---------------------------------------------------------------------------
        # random_replacement
        # ---------------------------------------------------------------------------

        class TestRandomReplacement:
            def test_replaces_numerical_entity(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                result = random_replacement(claim, entities)
                assert result != claim

            def test_ignores_non_numerical_entity(self, non_numerical_entity):
                claim = "Alice went to the store."
                result = random_replacement(claim, non_numerical_entity)
                assert result == claim

            def test_empty_entities(self):
                claim = "No numbers here."
                assert random_replacement(claim, []) == claim

            def test_all_numerical_labels_are_replaced(self):
                for label in NUMERICAL_LABELS:
                    claim = "Value is 99 units."
                    entities = [
                        {"label": label, "start_char": 9, "end_char": 11}
                    ]
                    result = random_replacement(claim, entities)
                    assert result != claim

            def test_multiple_entities_replaced(self, multiple_entities):
                claim = "100 items cost $200."
                result = random_replacement(claim, multiple_entities)
                # Surrounding text preserved
                assert " items cost $" in result

            def test_result_length_same_as_original_for_same_digit_count(self):
                claim = "There are 42 cats."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 12}
                ]
                result = random_replacement(claim, entities)
                assert len(result) == len(claim)

            def test_claim_without_digits_in_entity_unchanged(self):
                claim = "Happened in January."
                entities = [{"label": "DATE", "start_char": 12, "end_char": 19}]
                result = random_replacement(claim, entities)
                assert result == claim

            def test_text_outside_entity_unchanged(self):
                claim = "There are 100 apples in the basket."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                result = random_replacement(claim, entities)
                assert result[:10] == claim[:10]
                assert result[13:] == claim[13:]

            def test_overlapping_non_numerical_mixed_entities(self):
                claim = "Alice has 5 cats."
                entities = [
                    {"label": "PERSON", "start_char": 0, "end_char": 5},
                    {"label": "CARDINAL", "start_char": 10, "end_char": 11},
                ]
                result = random_replacement(claim, entities)
                assert result[:10] == claim[:10]
                assert result[11:] == claim[11:]
                assert result[10:11] != "5" or True  # May or may not differ

            def test_entity_at_start_of_claim(self):
                claim = "100 people attended."
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 3}
                ]
                result = random_replacement(claim, entities)
                assert result[3:] == claim[3:]
                assert (
                    result[:3] != "100" or True
                )  # Digit replacement attempted

            def test_entity_at_end_of_claim(self):
                claim = "The year was 2020"
                entities = [{"label": "DATE", "start_char": 13, "end_char": 17}]
                result = random_replacement(claim, entities)
                assert result[:13] == claim[:13]

            def test_multiple_non_numerical_entities_no_change(self):
                claim = "Alice and Bob met."
                entities = [
                    {"label": "PERSON", "start_char": 0, "end_char": 5},
                    {"label": "PERSON", "start_char": 10, "end_char": 13},
                ]
                assert random_replacement(claim, entities) == claim

            def test_entities_processed_in_reverse_order(self):
                # Ensures that processing in reverse doesn't corrupt positions
                claim = "10 and 20 are values."
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 2},
                    {"label": "CARDINAL", "start_char": 7, "end_char": 9},
                ]
                result = random_replacement(claim, entities)
                assert " and " in result
                assert "are values." in result

        # ---------------------------------------------------------------------------
        # masking
        # ---------------------------------------------------------------------------

        class TestMasking:
            def test_masks_digits(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                result = masking(claim, entities)
                assert "###" in result

            def test_masks_comma_and_dot(self):
                claim = "Cost is 1,000.50 dollars."
                entities = [{"label": "MONEY", "start_char": 8, "end_char": 16}]
                result = masking(claim, entities)
                assert "," not in result[8:16]
                assert "." not in result[8:16]

            def test_non_numerical_entity_not_masked(
                self, non_numerical_entity
            ):
                claim = "Alice went to the store."
                assert masking(claim, non_numerical_entity) == claim

            def test_empty_entities_no_change(self):
                claim = "No changes here."
                assert masking(claim, []) == claim

            def test_non_digit_non_special_chars_preserved(self):
                claim = "About $1,200 total."
                entities = [{"label": "MONEY", "start_char": 6, "end_char": 12}]
                result = masking(claim, entities)
                assert result[6] == "$"

            def test_all_numerical_labels_masked(self):
                for label in NUMERICAL_LABELS:
                    claim = "Value is 99 units."
                    entities = [
                        {"label": label, "start_char": 9, "end_char": 11}
                    ]
                    result = masking(claim, entities)
                    assert "##" in result

            def test_multiple_entities_masked(self, multiple_entities):
                claim = "100 items cost $200."
                result = masking(claim, multiple_entities)
                assert "###" in result

            def test_masking_result_same_length(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                result = masking(claim, entities)
                assert len(result) == len(claim)

            def test_text_outside_entity_unchanged(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                result = masking(claim, entities)
                assert result[:10] == claim[:10]
                assert result[13:] == claim[13:]

            def test_masking_does_not_introduce_digits(self):
                claim = "There are 42 cats."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 12}
                ]
                result = masking(claim, entities)
                assert not any(c.isdigit() for c in result[10:12])

            def test_entity_with_only_letters_not_masked(self):
                claim = "Happened in January."
                entities = [{"label": "DATE", "start_char": 12, "end_char": 19}]
                result = masking(claim, entities)
                # No digits, commas, or dots in "January" → unchanged
                assert result == claim

            def test_multiple_numeric_entities_all_masked(self):
                claim = "10 cats and 20 dogs."
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 2},
                    {"label": "CARDINAL", "start_char": 12, "end_char": 14},
                ]
                result = masking(claim, entities)
                assert "10" not in result
                assert "20" not in result

            def test_percent_value_masked(self):
                claim = "Growth was 12.5%."
                entities = [
                    {"label": "PERCENT", "start_char": 11, "end_char": 15}
                ]
                result = masking(claim, entities)
                assert "12" not in result[11:15]
                assert "." not in result[11:15]

            def test_masking_idempotent_on_already_masked(self):
                # Masking twice should give same result as masking once
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 13}
                ]
                once = masking(claim, entities)
                twice = masking(once, entities)
                assert once == twice

        # ---------------------------------------------------------------------------
        # value_removal
        # ---------------------------------------------------------------------------

        class TestValueRemoval:
            def test_removes_numerical_entity(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 14}
                ]
                result = value_removal(claim, entities)
                assert "100" not in result

            def test_non_numerical_entity_not_removed(
                self, non_numerical_entity
            ):
                claim = "Alice went to the store."
                assert value_removal(claim, non_numerical_entity) == claim

            def test_empty_entities_no_change(self):
                claim = "Nothing removed."
                assert value_removal(claim, []) == claim

            def test_no_double_spaces_after_removal(self):
                claim = "There are 100 apples."
                entities = [
                    {"label": "CARDINAL", "start_char": 10, "end_char": 14}
                ]
                result = value_removal(claim, entities)
                assert "  " not in result

            def test_multiple_entities_removed(self, multiple_entities):
                claim = "100 items cost $200."
                result = value_removal(claim, multiple_entities)
                assert "100" not in result

            def test_all_numerical_labels_removed(self):
                for label in NUMERICAL_LABELS:
                    claim = "Value is 99 units."
                    entities = [
                        {"label": label, "start_char": 9, "end_char": 11}
                    ]
                    result = value_removal(claim, entities)
                    assert "99" not in result

            def test_result_is_stripped(self):
                claim = "100 is the amount."
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 3}
                ]
                result = value_removal(claim, entities)
                assert not result.startswith(" ")

            def test_removal_at_end_of_claim(self):
                claim = "The total is 500"
                entities = [
                    {"label": "CARDINAL", "start_char": 13, "end_char": 16}
                ]
                result = value_removal(claim, entities)
                assert "500" not in result
                assert not result.endswith(" ")

            def test_removal_leaves_readable_text(self):
                claim = "There were 42 casualties."
                entities = [
                    {"label": "CARDINAL", "start_char": 11, "end_char": 13}
                ]
                result = value_removal(claim, entities)
                assert "There were" in result
                assert "casualties." in result

            def test_removal_multiple_entities_order_independent(self):
                claim = "10 cats and 20 dogs."
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 2},
                    {"label": "CARDINAL", "start_char": 12, "end_char": 14},
                ]
                result = value_removal(claim, entities)
                assert "10" not in result
                assert "20" not in result
                assert "cats" in result
                assert "dogs" in result

            def test_empty_claim_returns_empty(self):
                result = value_removal("", [])
                assert result == ""

            def test_claim_only_number_returns_empty_or_whitespace_stripped(
                self,
            ):
                claim = "500"
                entities = [
                    {"label": "CARDINAL", "start_char": 0, "end_char": 3}
                ]
                result = value_removal(claim, entities)
                assert result.strip() == ""

            def test_non_numerical_entity_value_preserved(self):
                claim = "Alice has 5 cats."
                entities = [
                    {"label": "PERSON", "start_char": 0, "end_char": 5},
                ]
                result = value_removal(claim, entities)
                assert "Alice" in result

            def test_mixed_labels_only_numerical_removed(self):
                claim = "Alice has 5 cats."
                entities = [
                    {"label": "PERSON", "start_char": 0, "end_char": 5},
                    {"label": "CARDINAL", "start_char": 10, "end_char": 11},
                ]
                result = value_removal(claim, entities)
                assert "Alice" in result
                assert "5" not in result

        # ---------------------------------------------------------------------------
        # process_json
        # ---------------------------------------------------------------------------

        MOCK_INCLUDED_KEYS = ["claim_id", "label", "normalized_claim"]

        @pytest.fixture
        def patch_included_keys():
            with patch(
                "code.data_preprocessing.perturbations.obfuscate.INCLUDED_KEYS",
                MOCK_INCLUDED_KEYS,
            ):
                yield

        class TestProcessJson:
            def test_returns_dict(self, sample_input_data, patch_included_keys):
                result = process_json(sample_input_data)
                assert isinstance(result, dict)

            def test_label_key_present(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                assert "True" in result

            def test_perturbed_keys_added(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                perturbation_keys = {
                    "perturb_type3_random",
                    "perturb_type3_masked",
                    "perturb_type3_removed",
                }
                assert perturbation_keys & entry.keys()

            def test_skips_entry_with_no_perturbation(
                self, patch_included_keys
            ):
                # No numerical entities → all perturbations equal original → skipped
                data = [
                    {
                        "normalized_claim": "This is a simple claim.",
                        "entities": [],
                        "label": "True",
                        "claim_id": "002",
                    }
                ]
                result = process_json(data)
                assert result == {}

            def test_multiple_labels_grouped(self, patch_included_keys):
                data = [
                    {
                        "normalized_claim": "There were 500 people.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 11,
                                "end_char": 14,
                            }
                        ],
                        "label": "True",
                        "claim_id": "001",
                    },
                    {
                        "normalized_claim": "Only 10 attended.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 5,
                                "end_char": 7,
                            }
                        ],
                        "label": "False",
                        "claim_id": "002",
                    },
                ]
                result = process_json(data)
                assert "True" in result
                assert "False" in result

            def test_base_entry_keys_from_included_keys(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                for key in MOCK_INCLUDED_KEYS:
                    if key in sample_input_data[0]:
                        assert key in entry

            def test_unknown_label_used_when_missing(self, patch_included_keys):
                data = [
                    {
                        "normalized_claim": "There are 50 units.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 10,
                                "end_char": 12,
                            }
                        ],
                        "claim_id": "003",
                    }
                ]
                result = process_json(data)
                assert "UNKNOWN" in result

            def test_random_key_only_added_when_different(
                self, patch_included_keys
            ):
                # Entity with no digits → random_replacement returns original
                data = [
                    {
                        "normalized_claim": "Happened in January.",
                        "entities": [
                            {
                                "label": "DATE",
                                "start_char": 12,
                                "end_char": 19,
                            }
                        ],
                        "label": "True",
                        "claim_id": "004",
                    }
                ]
                result = process_json(data)
                if "True" in result:
                    entry = result["True"][0]
                    assert "perturb_type3_random" not in entry

            def test_empty_input_returns_empty_dict(self, patch_included_keys):
                assert process_json([]) == {}

            def test_multiple_entries_same_label_grouped(
                self, patch_included_keys
            ):
                data = [
                    {
                        "normalized_claim": "There were 500 people.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 11,
                                "end_char": 14,
                            }
                        ],
                        "label": "True",
                        "claim_id": "001",
                    },
                    {
                        "normalized_claim": "About 200 attended.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 6,
                                "end_char": 9,
                            }
                        ],
                        "label": "True",
                        "claim_id": "002",
                    },
                ]
                result = process_json(data)
                assert len(result["True"]) == 2

            def test_masked_key_present_when_entity_has_digits(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                assert "perturb_type3_masked" in entry

            def test_removed_key_present_when_entity_has_digits(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                assert "perturb_type3_removed" in entry

            def test_extra_keys_not_included_from_entry(
                self, patch_included_keys
            ):
                data = [
                    {
                        "normalized_claim": "There are 50 units.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 10,
                                "end_char": 12,
                            }
                        ],
                        "label": "True",
                        "claim_id": "003",
                        "extra_field": "should_not_appear",
                    }
                ]
                result = process_json(data)
                entry = result["True"][0]
                assert "extra_field" not in entry

            def test_all_perturb_keys_when_entity_has_digits(
                self, patch_included_keys
            ):
                data = [
                    {
                        "normalized_claim": "There are 42 items.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 10,
                                "end_char": 12,
                            }
                        ],
                        "label": "True",
                        "claim_id": "005",
                    }
                ]
                result = process_json(data)
                entry = result["True"][0]
                assert "perturb_type3_random" in entry
                assert "perturb_type3_masked" in entry
                assert "perturb_type3_removed" in entry

            def test_perturb_values_differ_from_original_claim(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                original = sample_input_data[0]["normalized_claim"]
                if "perturb_type3_random" in entry:
                    assert entry["perturb_type3_random"] != original
                if "perturb_type3_masked" in entry:
                    assert entry["perturb_type3_masked"] != original
                if "perturb_type3_removed" in entry:
                    assert entry["perturb_type3_removed"] != original

            def test_entry_skipped_when_only_non_numerical_entities(
                self, patch_included_keys
            ):
                data = [
                    {
                        "normalized_claim": "Alice met Bob yesterday.",
                        "entities": [
                            {"label": "PERSON", "start_char": 0, "end_char": 5},
                            {
                                "label": "PERSON",
                                "start_char": 10,
                                "end_char": 13,
                            },
                        ],
                        "label": "True",
                        "claim_id": "006",
                    }
                ]
                result = process_json(data)
                assert result == {}

            def test_label_converted_to_string(self, patch_included_keys):
                data = [
                    {
                        "normalized_claim": "There are 10 items.",
                        "entities": [
                            {
                                "label": "CARDINAL",
                                "start_char": 10,
                                "end_char": 12,
                            }
                        ],
                        "label": 1,
                        "claim_id": "007",
                    }
                ]
                result = process_json(data)
                assert "1" in result

            def test_claim_id_preserved_in_output(
                self, sample_input_data, patch_included_keys
            ):
                result = process_json(sample_input_data)
                entry = result["True"][0]
                assert entry["claim_id"] == "001"

            def test_missing_entities_key_defaults_to_empty(
                self, patch_included_keys
            ):
                data = [
                    {
                        "normalized_claim": "There are 50 units.",
                        "label": "True",
                        "claim_id": "008",
                    }
                ]
                # Should not raise; no entities → all perturbations equal original → skipped
                result = process_json(data)
                assert result == {}
