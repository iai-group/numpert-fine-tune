"""Test suite for range perturbations in numerical claims preprocessing."""

import pytest
from src.data_preprocessing.perturbations import range as rng


def test_process_json_adds_perturbation_method():
    """Test process_json annotates emitted entries with perturbation type."""
    result = rng.process_json(
        [
            {
                "normalized_claim": "The rate is 50%.",
                "entities": [
                    {
                        "text": "50%",
                        "label": "PERCENT",
                        "start_char": 12,
                        "end_char": 15,
                    }
                ],
                "label": "True",
                "claim_id": "r1",
                "tokens": ["The", "rate", "is", "50", "%"],
                "ner_tags": ["O", "O", "O", "PERCENT", "PERCENT"],
            }
        ]
    )

    assert len(result) == 1
    assert result[0]["perturbation_method"] == "range"
    assert result[0]["claim_id"] == "r1"
    assert "entities" not in result[0]
    assert "tokens" not in result[0]
    assert "ner_tags" not in result[0]


def test_extract_number_from_text_percent():
    """Test extracting a number with a percent unit from text."""
    prefix, number, suffix, unit = rng.extract_number_from_text(
        "approx. 12.5 percent growth", "PERCENT"
    )
    assert prefix == "approx. "
    assert number == "12.5"
    assert suffix == " growth"
    assert unit.lower() == "percent"


def test_extract_number_from_text_money_with_symbol():
    """Test extracting a number with a currency symbol from text."""
    prefix, number, suffix, unit = rng.extract_number_from_text(
        "$1,234 revenue", "MONEY"
    )
    assert prefix == ""
    assert number == "1,234"
    assert suffix == " revenue"
    assert unit == "$"


def test_range_perturbation_handles_currency_prefix_even_if_cardinal():
    """Test range perturbation with a currency prefix."""
    out = rng.range_perturbation("$500", "CARDINAL", truth_value="true")
    assert out == "between $450 and $550"


@pytest.mark.parametrize(
    "text,entity_type,truth_value,expected",
    [
        ("50%", "PERCENT", "true", "between 45% and 55%"),
        ("50%", "PERCENT", "false", "between 75% and 80%"),
        (
            "around 33 percent",
            "PERCENT",
            "true",
            "around between 29 percent and 37 percent",
        ),
    ],
)
def test_perturb_percent(text, entity_type, truth_value, expected):
    """Test range perturbation for percent entities."""
    assert rng.range_perturbation(text, entity_type, truth_value) == expected


def test_perturb_time_supported_unit_true():
    """Test range perturbation for time with a supported unit."""
    out = rng.range_perturbation("3 hours", "TIME", "true")
    assert out == "between 2 and 4 hours"


def test_perturb_time_unsupported_unit_unmodified():
    """Test range perturbation for time with an unsupported unit."""
    assert rng.range_perturbation("2 weeks", "TIME", "true") == "2 weeks"


def test_perturb_money_true_and_false():
    """Test range perturbation for money with true and false values."""
    assert (
        rng.range_perturbation("$1000", "MONEY", "true")
        == "between $900 and $1,100"
    )
    assert (
        rng.range_perturbation("$1000", "MONEY", "false")
        == "between $1,150 and $1,250"
    )


def test_perturb_cardinal_true_and_false():
    """Test range perturbation for cardinal numbers with true/false values."""
    assert (
        rng.range_perturbation("1000", "CARDINAL", "true")
        == "between 900 and 1,100"
    )
    assert (
        rng.range_perturbation("1000", "CARDINAL", "false")
        == "between 1,450 and 1,650"
    )


def test_get_ordinal():
    """Test ordinal number formatting."""
    assert rng.get_ordinal(1) == "1st"
    assert rng.get_ordinal(2) == "2nd"
    assert rng.get_ordinal(3) == "3rd"
    assert rng.get_ordinal(4) == "4th"
    assert rng.get_ordinal(11) == "11th"
    assert rng.get_ordinal(12) == "12th"
    assert rng.get_ordinal(13) == "13th"
    assert rng.get_ordinal(21) == "21st"


def test_format_number_and_currency():
    """Test formatting numbers and currencies."""
    assert rng.format_number(1000.0) == "1,000"
    assert rng.format_number(1000.1) == "1,000.1"
    assert rng.format_currency(1000.0, "$") == "$1,000"
    assert rng.format_currency(1000.12, "$") == "$1,000.12"


def test_range_perturbation_returns_original_when_no_number_found():
    """Test perturbation returns original text when no number is found."""
    assert rng.range_perturbation("abc", "CARDINAL", "true") == "abc"
    assert rng.range_perturbation("abc", "MONEY", "true") == "abc"
    assert rng.range_perturbation("abc", "PERCENT", "true") == "abc"
    assert rng.range_perturbation("abc", "TIME", "true") == "abc"
    assert rng.range_perturbation("abc", "CARDINAL", "true") == "abc"


def test_perturb_claim_context_currency_symbol_and_capitalization():
    """Test claim with context, currency symbol, and capitalization."""
    claim = "He paid $500 for it."
    start = claim.index("500")
    end = start + len("500")
    entities = [
        {"text": "500", "label": "MONEY", "start_char": start, "end_char": end}
    ]
    out_true = rng.perturb_claim(claim, entities, "true")
    assert out_true == "He paid between $450 and $550 for it."

    # Capitalize when the perturbed span is at the start
    claim2 = "50% of users agreed."
    start2 = 0
    end2 = len("50%")
    entities2 = [
        {
            "text": "50%",
            "label": "PERCENT",
            "start_char": start2,
            "end_char": end2,
        }
    ]
    out2 = rng.perturb_claim(claim2, entities2, "true")
    assert out2.startswith("Between 45% and 55%")


# Tests for Indian monetary denominations (lakh and crore)


def test_range_perturbation_lakh_cardinal_true():
    """Test that lakh is preserved in range perturbation for CARDINAL type."""
    result = rng.range_perturbation("5 lakh", "CARDINAL", "true")
    assert "between" in result.lower()
    assert "lakh" in result.lower()


def test_range_perturbation_lakh_cardinal_false():
    """Test that lakh is preserved in false range perturbation for CARDINAL."""
    result = rng.range_perturbation("5 lakh", "CARDINAL", "false")
    assert "between" in result.lower()
    assert "lakh" in result.lower()


def test_range_perturbation_crore_cardinal_true():
    """Test that crore is preserved in range perturbation for CARDINAL type."""
    result = rng.range_perturbation("2 crore", "CARDINAL", "true")
    assert "between" in result.lower()
    assert "crore" in result.lower()


def test_range_perturbation_crore_cardinal_false():
    """Test that crore is preserved in false range perturbation for CARDINAL."""
    result = rng.range_perturbation("2 crore", "CARDINAL", "false")
    assert "between" in result.lower()
    assert "crore" in result.lower()


def test_range_perturbation_rupee_lakh_money_true():
    """Test that ₹ and lakh are preserved in MONEY range perturbation."""
    result = rng.range_perturbation("₹5 lakh", "MONEY", "true")
    assert "between" in result.lower()
    assert "₹" in result
    assert "lakh" in result.lower()


def test_range_perturbation_rupee_lakh_money_false():
    """Test that ₹ and lakh are preserved in false MONEY range perturbation."""
    result = rng.range_perturbation("₹5 lakh", "MONEY", "false")
    assert "between" in result.lower()
    assert "₹" in result
    assert "lakh" in result.lower()


def test_range_perturbation_rupee_crore_money_true():
    """Test that ₹ and crore are preserved in MONEY range perturbation."""
    result = rng.range_perturbation("₹2 crore", "MONEY", "true")
    assert "between" in result.lower()
    assert "₹" in result
    assert "crore" in result.lower()


def test_range_perturbation_rupee_crore_money_false():
    """Test that ₹ and crore are preserved in false MONEY range perturbation."""
    result = rng.range_perturbation("₹2 crore", "MONEY", "false")
    assert "between" in result.lower()
    assert "₹" in result
    assert "crore" in result.lower()


def test_perturb_claim_with_lakh():
    """Test claim perturbation preserves lakh denomination."""
    claim = "The property costs 10 lakh rupees."
    start = claim.index("10")
    end = start + len("10 lakh")
    entities = [
        {
            "text": "10 lakh",
            "label": "CARDINAL",
            "start_char": start,
            "end_char": end,
        }
    ]
    result = rng.perturb_claim(claim, entities, "true")
    assert "lakh" in result.lower()
    assert "between" in result.lower()


def test_perturb_claim_with_crore():
    """Test claim perturbation preserves crore denomination."""
    claim = "The budget allocation is 5 crore rupees."
    start = claim.index("5")
    end = start + len("5 crore")
    entities = [
        {
            "text": "5 crore",
            "label": "CARDINAL",
            "start_char": start,
            "end_char": end,
        }
    ]
    result = rng.perturb_claim(claim, entities, "true")
    assert "crore" in result.lower()
    assert "between" in result.lower()


def test_perturb_claim_with_rupee_lakh():
    """Test claim perturbation with rupee symbol and lakh."""
    claim = "He invested ₹25 lakh in company shares."
    start = claim.index("₹")
    end = claim.index("lakh") + len("lakh")
    entities = [
        {
            "text": "₹25 lakh",
            "label": "MONEY",
            "start_char": start,
            "end_char": end,
        }
    ]
    result = rng.perturb_claim(claim, entities, "true")
    assert "₹" in result
    assert "lakh" in result.lower()
    assert "between" in result.lower()


def test_perturb_claim_with_rupee_crore():
    """Test claim perturbation with rupee symbol and crore."""
    claim = "The donation was ₹15 crore to the hospital."
    start = claim.index("₹")
    end = claim.index("crore") + len("crore")
    entities = [
        {
            "text": "₹15 crore",
            "label": "MONEY",
            "start_char": start,
            "end_char": end,
        }
    ]
    result = rng.perturb_claim(claim, entities, "true")
    assert "₹" in result
    assert "crore" in result.lower()
    assert "between" in result.lower()


@pytest.mark.parametrize(
    "text,entity_type,truth_value,expected_unit",
    [
        ("5 lakh", "CARDINAL", "true", "lakh"),
        ("5 lakh", "CARDINAL", "false", "lakh"),
        ("10 lakh", "CARDINAL", "true", "lakh"),
        ("10 lakh", "CARDINAL", "false", "lakh"),
        ("2 crore", "CARDINAL", "true", "crore"),
        ("2 crore", "CARDINAL", "false", "crore"),
        ("50 crore", "CARDINAL", "true", "crore"),
        ("50 crore", "CARDINAL", "false", "crore"),
        ("₹5 lakh", "MONEY", "true", "lakh"),
        ("₹5 lakh", "MONEY", "false", "lakh"),
        ("₹1 crore", "MONEY", "true", "crore"),
        ("₹1 crore", "MONEY", "false", "crore"),
    ],
)
def test_range_perturbation_indian_denominations_preserve_unit(
    text, entity_type, truth_value, expected_unit
):
    """Parametrized test to verify Indian monetary units are preserved."""
    result = rng.range_perturbation(text, entity_type, truth_value)
    assert expected_unit.lower() in result.lower(), (
        f"Expected '{expected_unit}' in result '{result}' "
        f"for input '{text}' with type '{entity_type}'"
    )
    assert (
        "between" in result.lower()
    ), f"Expected 'between' in perturbed result '{result}'"


def test_perturb_cardinal_small_number():
    """Test cardinal perturbation with a small number rounds to zero lower
    bound."""
    out = rng.range_perturbation("5", "CARDINAL", "true")
    assert out in ("between 4 and 6", "between 5 and 6")


def test_is_integer_number_with_int():
    """Test _is_integer_number with integer input."""
    assert rng._is_integer_number(5) is True
    assert rng._is_integer_number(0) is True
    assert rng._is_integer_number(-10) is True


def test_is_integer_number_with_float_integer_value():
    """Test _is_integer_number with float that represents an integer."""
    assert rng._is_integer_number(5.0) is True
    assert rng._is_integer_number(0.0) is True
    assert rng._is_integer_number(-10.0) is True


def test_is_integer_number_with_float_decimal_value():
    """Test _is_integer_number with float that has decimal part."""
    assert rng._is_integer_number(5.5) is False
    assert rng._is_integer_number(0.1) is False
    assert rng._is_integer_number(-10.25) is False
    assert rng._is_integer_number(1.0001) is False


def test_is_integer_number_with_other_type():
    """Test _is_integer_number with unsupported type."""
    assert rng._is_integer_number("5") is False
    assert rng._is_integer_number(None) is False
    assert rng._is_integer_number([5]) is False
