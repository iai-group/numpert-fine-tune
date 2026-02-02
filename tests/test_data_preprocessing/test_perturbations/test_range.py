"""Test suite for range perturbations in numerical claims preprocessing."""

import pytest
from src.data_preprocessing.perturbations import range as rng


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
            "around between 29.7 percent and 36.3 percent",
        ),
    ],
)
def test_perturb_percent(text, entity_type, truth_value, expected):
    """Test range perturbation for percent entities."""
    assert rng.range_perturbation(text, entity_type, truth_value) == expected


def test_perturb_time_supported_unit_true():
    """Test range perturbation for time with a supported unit."""
    out = rng.range_perturbation("3 hours", "TIME", "true")
    assert out == "between 2.7 and 3.3 hours"


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
