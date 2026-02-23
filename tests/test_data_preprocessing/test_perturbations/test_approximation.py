"""Tests for the approximation perturbation module."""
import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from code.data_preprocessing.perturbations.approximation import (
    approximate_number,
    apply_numeric_spans,
    build_token_char_map,
    contains_approximation,
    exaggerate_number,
    expand_span_to_include_currency,
    extract_numeric_spans,
    find_numeric_token_spans,
    format_like,
    is_sentence_start,
    needs_about,
    NumericSpan,
    parse_number,
    perturb_claim,
    process_claims,
    TARGET_LABELS,
    APPROXIMATION_KEYWORDS,
)


# =========================
# Configuration Tests
# =========================


class TestConfiguration:
    """Tests for approximation module configuration."""

    def test_target_labels_contains_expected(self):
        """Verifies TARGET_LABELS contains expected entity types."""
        expected = {"CARDINAL", "PERCENT", "MONEY", "QUANTITY", "TIME"}
        assert TARGET_LABELS == expected

    def test_approximation_keywords_contains_about(self):
        """Verifies APPROXIMATION_KEYWORDS contains 'about'."""
        assert "about" in APPROXIMATION_KEYWORDS

    def test_approximation_keywords_contains_around(self):
        """Verifies APPROXIMATION_KEYWORDS contains 'around'."""
        assert "around" in APPROXIMATION_KEYWORDS

    def test_approximation_keywords_nonempty(self):
        """Verifies APPROXIMATION_KEYWORDS is not empty."""
        assert len(APPROXIMATION_KEYWORDS) > 0


# =========================
# NumericSpan Tests
# =========================


class TestNumericSpan:
    """Tests for NumericSpan dataclass."""

    def test_creation(self):
        """Verifies NumericSpan can be created with expected attributes."""
        span = NumericSpan(
            token_start=0,
            token_end=1,
            char_start=0,
            char_end=3,
            number_token_idx=0,
            original_number="100",
            original_value=100.0,
            approx_true=100.0,
            approx_false=200.0,
        )
        assert span.original_number == "100"
        assert span.original_value == 100.0
        assert span.approx_true == 100.0
        assert span.approx_false == 200.0


# =========================
# approximate_number Tests
# =========================


class TestApproximateNumber:
    """Tests for approximate_number function."""

    def test_zero_returns_zero(self):
        """Verifies zero returns zero when approximated."""
        assert approximate_number(0) == 0

    def test_small_number_rounds_normally(self):
        """Verifies small numbers round unchanged."""
        assert approximate_number(7) == 7
        assert approximate_number(3) == 3

    def test_small_number_rounds_to_int(self):
        """Verifies floats are correctly rounded to integers."""
        assert approximate_number(4.6) == 5
        assert approximate_number(2.3) == 2

    def test_already_round_number_unchanged(self):
        """Verifies already-round numbers remain unchanged."""
        assert approximate_number(100) == 100
        assert approximate_number(1000) == 1000

    def test_large_number_snaps_to_step(self):
        """Verifies large numbers snap to expected step value."""
        result = approximate_number(13000)
        assert result in [10000, 12000, 15000]

    def test_negative_number(self):
        """Verifies negative numbers are handled correctly."""
        result = approximate_number(-100)
        assert result == -100

    def test_negative_small_number(self):
        """Verifies small negative numbers remain unchanged."""
        result = approximate_number(-7)
        assert result == -7

    def test_returns_float_or_int(self):
        """Verifies return type is int or float."""
        result = approximate_number(5000)
        assert isinstance(result, (int, float))

    def test_already_on_step_1200(self):
        """Verifies 1200 remains unchanged as already on step."""
        result = approximate_number(1200)
        assert result == 1200

    def test_already_on_step_2500(self):
        """Verifies 2500 remains unchanged as already on step."""
        result = approximate_number(2500)
        assert result == 2500


# =========================
# exaggerate_number Tests
# =========================


class TestExaggerateNumber:
    """Tests for exaggerate_number function."""

    def test_zero_returns_nonzero(self):
        """Verifies zero returns a non-zero exaggerated value."""
        result = exaggerate_number(0)
        assert result in [10.0, 50.0]

    def test_small_positive_returns_double_digit(self):
        """Verifies small positive numbers exaggerate to double digit."""
        result = exaggerate_number(5)
        assert result in [10, 12, 15, 20, 25, 50]

    def test_small_negative_returns_negative_double_digit(self):
        """Verifies small negatives exaggerate to negative double digit."""
        result = exaggerate_number(-5)
        assert result in [-10, -12, -15, -20, -25, -50]

    def test_large_number_differs_from_original(self):
        """Verifies large numbers exaggerate to different value."""
        result = exaggerate_number(1000)
        assert result != 1000

    def test_large_number_positive(self):
        """Verifies large positive number exaggerates to positive."""
        result = exaggerate_number(10000)
        assert result > 0

    @patch(
        "code.data_preprocessing.perturbations.approximation"
        ".random.choice"
    )
    def test_zero_uses_random_choice(self, mock_choice):
        """Verifies zero exaggeration uses random.choice."""
        mock_choice.return_value = 10.0
        result = exaggerate_number(0)
        assert result == 10.0


# =========================
# parse_number Tests
# =========================


class TestParseNumber:
    """Tests for parse_number function."""

    def test_plain_integer(self):
        """Verifies plain integers are correctly parsed."""
        assert parse_number("100") == 100.0

    def test_float(self):
        """Verifies floats are correctly parsed."""
        assert parse_number("3.14") == 3.14

    def test_comma_separated(self):
        """Verifies comma-separated numbers are parsed correctly."""
        assert parse_number("1,000") == 1000.0

    def test_invalid_returns_none(self):
        """Verifies invalid input returns None."""
        assert parse_number("abc") is None

    def test_empty_string_returns_none(self):
        """Verifies empty string returns None."""
        assert parse_number("") is None

    def test_large_comma_number(self):
        """Verifies large comma-separated numbers are parsed."""
        assert parse_number("1,000,000") == 1000000.0


# =========================
# format_like Tests
# =========================


class TestFormatLike:
    """Tests for format_like function."""

    def test_no_comma_plain(self):
        """Verifies plain number is formatted without comma."""
        assert format_like("100", 200.0) == "200"

    def test_with_comma_formats_with_comma(self):
        """Verifies comma format is preserved in output."""
        result = format_like("1,000", 2000.0)
        assert "," in result
        assert result == "2,000"

    def test_rounds_to_int(self):
        """Verifies value is rounded to nearest integer."""
        assert format_like("10", 10.7) == "11"

    def test_large_number_with_comma(self):
        """Verifies large numbers are formatted with commas."""
        result = format_like("10,000", 20000.0)
        assert result == "20,000"


# =========================
# find_numeric_token_spans Tests
# =========================


class TestFindNumericTokenSpans:
    """Tests for find_numeric_token_spans function."""

    def test_single_cardinal(self):
        """Verifies single CARDINAL token is correctly identified."""
        tokens = ["There", "were", "100", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        spans = find_numeric_token_spans(tokens, ner_tags)
        assert spans == [(2, 3)]

    def test_no_numeric_tokens(self):
        """Verifies empty list when no numeric tokens found."""
        tokens = ["Hello", "world"]
        ner_tags = ["O", "O"]
        result = find_numeric_token_spans(tokens, ner_tags)
        assert result == []

    def test_consecutive_same_label(self):
        """Verifies consecutive same labels are merged into span."""
        tokens = ["one", "hundred"]
        ner_tags = ["CARDINAL", "CARDINAL"]
        spans = find_numeric_token_spans(tokens, ner_tags)
        assert spans == [(0, 2)]

    def test_multiple_spans(self):
        """Verifies multiple non-consecutive spans are identified."""
        tokens = ["100", "and", "200"]
        ner_tags = ["CARDINAL", "O", "CARDINAL"]
        spans = find_numeric_token_spans(tokens, ner_tags)
        assert spans == [(0, 1), (2, 3)]

    def test_money_label(self):
        """Verifies MONEY labels are correctly identified."""
        tokens = ["$", "50"]
        ner_tags = ["MONEY", "MONEY"]
        spans = find_numeric_token_spans(tokens, ner_tags)
        assert spans == [(0, 2)]


# =========================
# build_token_char_map Tests
# =========================


class TestBuildTokenCharMap:
    """Tests for build_token_char_map function."""

    def test_simple_sentence(self):
        """Verifies token character map is correctly built."""
        text = "100 people"
        tokens = ["100", "people"]
        spans = build_token_char_map(text, tokens)
        assert spans[0] == (0, 3)
        assert spans[1] == (4, 10)

    def test_token_not_found_returns_minus_one(self):
        """Verifies missing token returns (-1, -1)."""
        text = "hello world"
        tokens = ["hello", "xyz"]
        spans = build_token_char_map(text, tokens)
        assert spans[1] == (-1, -1)

    def test_empty_tokens(self):
        """Verifies empty token list returns empty spans."""
        spans = build_token_char_map("hello", [])
        assert spans == []


# =========================
# expand_span_to_include_currency Tests
# =========================


class TestExpandSpanToIncludeCurrency:
    """Tests for expand_span_to_include_currency function."""

    def test_dollar_sign(self):
        """Verifies dollar sign is included in span."""
        text = "$100"
        result = expand_span_to_include_currency(text, 1)
        assert result == 0

    def test_no_currency(self):
        """Verifies no change when no currency present."""
        text = "100 people"
        result = expand_span_to_include_currency(text, 0)
        assert result == 0

    def test_euro_sign(self):
        """Verifies euro sign is included in span."""
        text = "€200"
        result = expand_span_to_include_currency(text, 1)
        assert result == 0

    def test_usd_code(self):
        """Verifies USD code prefix is included in span."""
        text = "USD100"
        result = expand_span_to_include_currency(text, 3)
        assert result == 0

    def test_no_currency_returns_start(self):
        """Verifies start position returned when no currency."""
        text = "about 100"
        result = expand_span_to_include_currency(text, 6)
        assert result == 6


# =========================
# extract_numeric_spans Tests
# =========================


class TestExtractNumericSpans:
    """Tests for extract_numeric_spans function."""

    def test_extracts_single_span(self):
        """Verifies single numeric span is correctly extracted."""
        text = "There were 100 people"
        tokens = ["There", "were", "100", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        spans = extract_numeric_spans(text, tokens, ner_tags)
        assert len(spans) == 1
        assert spans[0].original_number == "100"
        assert spans[0].original_value == 100.0

    def test_no_numeric_returns_empty(self):
        """Verifies empty list when no numeric tokens found."""
        text = "No numbers here"
        tokens = ["No", "numbers", "here"]
        ner_tags = ["O", "O", "O"]
        spans = extract_numeric_spans(text, tokens, ner_tags)
        assert spans == []

    def test_span_with_no_parseable_number_skipped(self):
        """Verifies unparseable numbers are skipped."""
        text = "one hundred"
        tokens = ["one", "hundred"]
        ner_tags = ["CARDINAL", "CARDINAL"]
        spans = extract_numeric_spans(text, tokens, ner_tags)
        assert spans == []

    def test_sets_approx_true_and_false(self):
        """Verifies approximation values are set correctly."""
        text = "There were 1000 people"
        tokens = ["There", "were", "1000", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        spans = extract_numeric_spans(text, tokens, ner_tags)
        assert len(spans) == 1
        assert spans[0].approx_true is not None
        assert spans[0].approx_false is not None


# =========================
# needs_about Tests
# =========================


class TestNeedsAbout:
    """Tests for needs_about function."""

    def test_no_approximation_word_needs_about(self):
        """Verifies 'about' needed when no approximation present."""
        assert needs_about("There were") is True

    def test_already_has_about(self):
        """Verifies 'about' not needed when already present."""
        assert needs_about("There were about") is False

    def test_already_has_approximately(self):
        """Verifies 'about' not needed when 'approximately' present."""
        assert needs_about("There were approximately") is False

    def test_empty_prefix(self):
        """Verifies 'about' needed for empty string."""
        assert needs_about("") is True


# =========================
# is_sentence_start Tests
# =========================


class TestIsSentenceStart:
    """Tests for is_sentence_start function."""

    def test_empty_prefix(self):
        """Verifies empty string is considered sentence start."""
        assert is_sentence_start("") is True

    def test_ends_with_period(self):
        """Verifies text ending with period is sentence start."""
        assert is_sentence_start("End of sentence.") is True

    def test_ends_with_exclamation(self):
        """Verifies text ending with ! is sentence start."""
        assert is_sentence_start("Wow!") is True

    def test_mid_sentence(self):
        """Verifies mid-sentence text is not sentence start."""
        assert is_sentence_start("There were") is False

    def test_whitespace_only(self):
        """Verifies whitespace-only text is sentence start."""
        assert is_sentence_start("   ") is True


# =========================
# apply_numeric_spans Tests
# =========================


class TestApplyNumericSpans:
    """Tests for apply_numeric_spans function."""

    def _make_span(
        self,
        char_start,
        char_end,
        original_number,
        approx_true,
        approx_false,
    ):
        """Helper to create a NumericSpan for testing.

        Args:
            char_start: Character position where span starts.
            char_end: Character position where span ends.
            original_number: Original number string.
            approx_true: Value for true approximation.
            approx_false: Value for false approximation.

        Returns:
            NumericSpan: Constructed span with given parameters.
        """
        return NumericSpan(
            token_start=0,
            token_end=1,
            char_start=char_start,
            char_end=char_end,
            number_token_idx=0,
            original_number=original_number,
            original_value=float(original_number.replace(",", "")),
            approx_true=approx_true,
            approx_false=approx_false,
        )

    def test_true_uses_approx_true(self):
        """Verifies true variant uses approx_true value."""
        text = "There were 100 people"
        span = self._make_span(11, 14, "100", 100.0, 200.0)
        result = apply_numeric_spans(text, [span], is_true=True)
        assert "100" in result

    def test_false_uses_approx_false(self):
        """Verifies false variant uses approx_false value."""
        text = "There were 100 people"
        span = self._make_span(11, 14, "100", 100.0, 200.0)
        result = apply_numeric_spans(text, [span], is_true=False)
        assert "200" in result

    def test_about_added_at_sentence_start(self):
        """Verifies 'About' added at sentence start."""
        text = "100 people attended"
        span = self._make_span(0, 3, "100", 100.0, 200.0)
        result = apply_numeric_spans(text, [span], is_true=True)
        assert result.startswith("About")

    def test_about_added_mid_sentence(self):
        """Verifies 'about' added in middle of sentence."""
        text = "There were 100 people"
        span = self._make_span(11, 14, "100", 100.0, 200.0)
        result = apply_numeric_spans(text, [span], is_true=True)
        assert "about" in result.lower()

    def test_no_about_if_already_present(self):
        """Verifies 'about' not duplicated if already present."""
        text = "There were about 100 people"
        span = self._make_span(17, 20, "100", 100.0, 200.0)
        result = apply_numeric_spans(text, [span], is_true=True)
        assert result.lower().count("about") == 1


# =========================
# contains_approximation Tests
# =========================


class TestContainsApproximation:
    """Tests for contains_approximation function."""

    def test_contains_about(self):
        """Verifies 'about' keyword is detected."""
        text = "There were about 100 people"
        assert contains_approximation(text) is True

    def test_contains_approximately(self):
        """Verifies 'approximately' keyword is detected."""
        text = "Approximately 100 people"
        assert contains_approximation(text) is True

    def test_no_approximation(self):
        """Verifies no keywords detected in plain text."""
        text = "There were 100 people"
        assert contains_approximation(text) is False

    def test_case_insensitive(self):
        """Verifies detection is case-insensitive."""
        text = "ABOUT 100 people"
        assert contains_approximation(text) is True

    def test_empty_string(self):
        """Verifies empty string has no approximation keywords."""
        assert contains_approximation("") is False


# =========================
# perturb_claim Tests
# =========================


class TestPerturbClaim:
    """Tests for perturb_claim function."""

    def test_returns_none_when_no_spans(self):
        """Verifies None returned when no numeric spans found."""
        claim = "No numbers here"
        tokens = ["No", "numbers", "here"]
        ner_tags = ["O", "O", "O"]
        t, f = perturb_claim(claim, tokens, ner_tags)
        assert t is None
        assert f is None

    def test_returns_perturbed_claims(self):
        """Verifies perturbed claims are returned."""
        claim = "There were 1000 people"
        tokens = ["There", "were", "1000", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        t, f = perturb_claim(claim, tokens, ner_tags)
        assert t is not None
        assert f is not None

    def test_true_claim_differs_from_original(self):
        """Verifies true variant differs from original."""
        claim = "There were 1000 people"
        tokens = ["There", "were", "1000", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        t, _ = perturb_claim(claim, tokens, ner_tags)
        assert t != claim

    def test_false_claim_differs_from_original(self):
        """Verifies false variant differs from original."""
        claim = "There were 1000 people"
        tokens = ["There", "were", "1000", "people"]
        ner_tags = ["O", "O", "CARDINAL", "O"]
        _, f = perturb_claim(claim, tokens, ner_tags)
        assert f != claim


# =========================
# process_claims Tests
# =========================


class TestProcessClaims:
    """Tests for process_claims function."""

    def _make_claims(self):
        """Helper to create sample test claims.

        Returns:
            List[Dict]: List with single sample claim data.
        """
        return [
            {
                "normalized_claim": "There were 1000 people",
                "tokens": ["There", "were", "1000", "people"],
                "ner_tags": ["O", "O", "CARDINAL", "O"],
                "label": "True",
                "claim_id": "1",
            }
        ]

    def test_writes_output_file(self, tmp_path):
        """Verifies output file is created."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(
            json.dumps(self._make_claims()), encoding="utf-8"
        )
        process_claims(str(input_file), str(output_file))
        assert output_file.exists()

    def test_output_contains_perturbed_keys(self, tmp_path):
        """Verifies output contains expected perturbed keys."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(
            json.dumps(self._make_claims()), encoding="utf-8"
        )
        process_claims(str(input_file), str(output_file))
        result = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(result) > 0
        assert "perturb_type1_True" in result[0]
        assert "perturb_type2_True" in result[0]

    def test_excludes_claims_with_approximation(self, tmp_path):
        """Verifies claims with approximations are excluded."""
        claims = [
            {
                "normalized_claim": "There were about 1000 people",
                "tokens": ["There", "were", "about", "1000", "people"],
                "ner_tags": ["O", "O", "O", "CARDINAL", "O"],
                "label": "True",
                "claim_id": "2",
            }
        ]
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(json.dumps(claims), encoding="utf-8")
        process_claims(str(input_file), str(output_file))
        result = json.loads(output_file.read_text(encoding="utf-8"))
        assert result == []

    def test_excludes_tokens_ner_from_output(self, tmp_path):
        """Verifies tokens/NER tags excluded from output."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(
            json.dumps(self._make_claims()), encoding="utf-8"
        )
        process_claims(str(input_file), str(output_file))
        result = json.loads(output_file.read_text(encoding="utf-8"))
        if result:
            assert "tokens" not in result[0]
            assert "ner_tags" not in result[0]
            assert "entities" not in result[0]

    def test_empty_input(self, tmp_path):
        """Verifies empty input produces empty output."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(json.dumps([]), encoding="utf-8")
        process_claims(str(input_file), str(output_file))
        result = json.loads(output_file.read_text(encoding="utf-8"))
        assert result == []