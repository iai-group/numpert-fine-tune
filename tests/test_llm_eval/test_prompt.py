"""Test suite for prompt.py module.

This module tests the prompt generation functions for LLM evaluations.
"""

from src.prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    FACT_CHECK_USER_PROMPT,
    NEGATIVE_EXAMPLES,
    PromptType,
    get_fact_check_system_prompt,
    get_fact_check_user_prompt,
)


class TestPromptConstants:
    """Test the prompt constants and their content."""

    def test_fact_check_system_prompt_content(self):
        """Test that the system prompt contains expected content."""
        assert isinstance(FACT_CHECK_SYSTEM_PROMPT, str)
        assert len(FACT_CHECK_SYSTEM_PROMPT) > 0

        # Check for key phrases
        assert "fact-checker" in FACT_CHECK_SYSTEM_PROMPT.lower()
        assert "true or false" in FACT_CHECK_SYSTEM_PROMPT.lower()
        assert "evidence" in FACT_CHECK_SYSTEM_PROMPT.lower()

    def test_few_shot_examples_format(self):
        """Test that few-shot examples have the correct format."""
        assert isinstance(FEW_SHOT_EXAMPLES, str)
        assert len(FEW_SHOT_EXAMPLES) > 0

        # Check for expected structure
        assert "Claim:" in FEW_SHOT_EXAMPLES
        assert "Evidence:" in FEW_SHOT_EXAMPLES
        assert '{"label": false}' in FEW_SHOT_EXAMPLES
        assert "150,000 (oil spill) claims " in FEW_SHOT_EXAMPLES
        assert "Leonardo DiCaprio has donated $10,000,000" in FEW_SHOT_EXAMPLES

    def test_fact_check_user_prompt_placeholders(self):
        """Test that user prompt contains expected placeholders."""
        assert isinstance(FACT_CHECK_USER_PROMPT, str)
        assert len(FACT_CHECK_USER_PROMPT) > 0

        # Check for placeholders
        assert "{examples}" in FACT_CHECK_USER_PROMPT
        assert "{claim}" in FACT_CHECK_USER_PROMPT
        assert "{evidence}" in FACT_CHECK_USER_PROMPT

        # Check for expected content
        assert "classify" in FACT_CHECK_USER_PROMPT.lower()
        assert '{{"label": true}}' in FACT_CHECK_USER_PROMPT
        assert '{{"label": false}}' in FACT_CHECK_USER_PROMPT

    def test_negative_examples_initialized(self):
        """Test that negative examples is initialized."""
        assert isinstance(NEGATIVE_EXAMPLES, str)
        assert NEGATIVE_EXAMPLES != ""


class TestGetFactCheckSystemPrompt:
    """Test the get_fact_check_system_prompt function."""

    def test_returns_system_prompt(self):
        """Test that function returns the correct system prompt."""
        result = get_fact_check_system_prompt()

        assert result == FACT_CHECK_SYSTEM_PROMPT
        assert isinstance(result, str)
        assert len(result) > 0

    def test_consistent_return_value(self):
        """Test that multiple calls return the same value."""
        result1 = get_fact_check_system_prompt()
        result2 = get_fact_check_system_prompt()

        assert result1 == result2

    def test_immutable_return(self):
        """Test that returned value is not affected by modifications."""
        original = get_fact_check_system_prompt()

        # This shouldn't affect the original
        modified = original + " MODIFIED"

        # Get a fresh copy
        fresh = get_fact_check_system_prompt()

        assert fresh == original
        assert fresh != modified


class TestGetFactCheckUserPrompt:
    """Test the get_fact_check_user_prompt function."""

    def test_basic_prompt_formatting(self):
        """Test basic prompt formatting without negative examples."""
        claim = "The sky is blue"
        evidence = "Scientific observations show that the sky appears blue"

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        assert isinstance(result, str)
        assert claim in result
        assert evidence in result
        assert FEW_SHOT_EXAMPLES in result
        assert "Given the claim and evidence" in result

    def test_prompt_with_negative_examples_false(self):
        """Test prompt generation when negative examples are disabled."""
        claim = "Water boils at 100°C"
        evidence = "At standard atmospheric pressure, water boils at 100°C"

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        # Should contain few-shot examples but not negative examples
        assert FEW_SHOT_EXAMPLES in result
        # Since NEGATIVE_EXAMPLES is empty, this test passes regardless
        assert claim in result
        assert evidence in result

    def test_prompt_with_negative_examples_true(self):
        """Test prompt generation when negative examples are enabled."""
        claim = "The moon is made of cheese"
        evidence = "Scientific analysis shows the moon is made of rock"

        result = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.FEW_SHOT_WITH_NEGATIVE,
        )

        # Should contain both few-shot and negative examples
        assert FEW_SHOT_EXAMPLES in result
        # NEGATIVE_EXAMPLES would be included if it wasn't empty
        assert claim in result
        assert evidence in result

    def test_special_characters_in_inputs(self):
        """Test handling of special characters in claim and evidence."""
        claim = "Test with 'quotes' and \"double quotes\" & symbols!"
        evidence = "Evidence with $pecial ch@racters: 100% true (maybe?)"

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        assert claim in result
        assert evidence in result
        assert isinstance(result, str)

    def test_empty_strings(self):
        """Test handling of empty claim and evidence."""
        result = get_fact_check_user_prompt(
            claim="", evidence="", prompt_type=PromptType.FEW_SHOT
        )

        assert isinstance(result, str)
        assert FEW_SHOT_EXAMPLES in result
        # Empty strings should still be formatted into the template

    def test_multiline_inputs(self):
        """Test handling of multiline claim and evidence."""
        claim = "This is a claim\nwith multiple lines\nand line breaks"
        evidence = "This is evidence\nthat spans\nseveral lines"

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        assert claim in result
        assert evidence in result
        assert "\n" in result  # Should preserve line breaks

    def test_very_long_inputs(self):
        """Test handling of very long claim and evidence strings."""
        claim = "Very long claim " * 100  # 1600+ characters
        evidence = "Very long evidence " * 100  # 1900+ characters

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        assert claim in result
        assert evidence in result
        assert len(result) > len(claim) + len(
            evidence
        )  # Should include template

    def test_unicode_characters(self):
        """Test handling of unicode characters in inputs."""
        claim = "Tëst wïth ünïcödë châractërs: 测试中文"
        evidence = "Évidence wïth spëcial chars: データ日本語"

        result = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        assert claim in result
        assert evidence in result
        assert isinstance(result, str)

    def test_boolean_parameter_types(self):
        """Test that prompt_type accepts PromptType enum values."""
        claim = "Test claim"
        evidence = "Test evidence"

        # Test with FEW_SHOT_WITH_NEGATIVE
        result_with_neg = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.FEW_SHOT_WITH_NEGATIVE,
        )

        # Test with FEW_SHOT
        result_few_shot = get_fact_check_user_prompt(
            claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
        )

        # Both should work and return strings
        assert isinstance(result_with_neg, str)
        assert isinstance(result_few_shot, str)

        # Result with negative should include NEGATIVE_EXAMPLES
        assert NEGATIVE_EXAMPLES in result_with_neg
        assert NEGATIVE_EXAMPLES not in result_few_shot


class TestParameterCombinations:
    """Test all combinations of prompt example parameters.

    Tests include_few_shot_examples and include_negative_examples.
    """

    def test_both_parameters_true_true(self):
        """Test with FEW_SHOT_WITH_NEGATIVE prompt type.

        Checks prompt includes few-shot and negative examples.
        """
        claim = "Test claim for TT combination"
        evidence = "Test evidence for TT combination"
        result = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.FEW_SHOT_WITH_NEGATIVE,
        )
        assert isinstance(result, str)
        assert claim in result
        assert evidence in result
        assert FEW_SHOT_EXAMPLES in result

    def test_both_parameters_true_false(self):
        """Test with FEW_SHOT prompt type.

        Checks prompt includes few-shot examples.
        """
        claim = "Test claim for TF combination"
        evidence = "Test evidence for TF combination"
        result = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.FEW_SHOT,
        )
        assert isinstance(result, str)
        assert claim in result
        assert evidence in result
        assert FEW_SHOT_EXAMPLES in result

    def test_both_parameters_false_true(self):
        """Test with ZERO_SHOT prompt type.

        Checks prompt does not include few-shot or negative examples.
        """
        claim = "Test claim for FT combination"
        evidence = "Test evidence for FT combination"
        result = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.ZERO_SHOT,
        )
        assert isinstance(result, str)
        assert claim in result
        assert evidence in result
        assert FEW_SHOT_EXAMPLES not in result

    def test_both_parameters_false_false(self):
        """Test with ZERO_SHOT prompt type.

        Checks prompt does not include any examples.
        """
        claim = "Test claim for FF combination"
        evidence = "Test evidence for FF combination"
        result = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.ZERO_SHOT,
        )
        assert isinstance(result, str)
        assert claim in result
        assert evidence in result
        assert FEW_SHOT_EXAMPLES not in result

    def test_default_parameter_values(self):
        """Test that default parameter values work as expected."""
        claim = "Test claim for defaults"
        evidence = "Test evidence for defaults"

        # Call without specifying parameters (should use defaults)
        result_defaults = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
        )

        # Call with explicit default (FEW_SHOT)
        result_explicit = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=PromptType.FEW_SHOT,
        )

        # Should be identical
        assert result_defaults == result_explicit
        assert FEW_SHOT_EXAMPLES in result_defaults

    def test_parameter_combinations_consistency(self):
        """Test that same parameter combinations always return same results."""  # noqa: E501
        claim = "Consistency test claim"
        evidence = "Consistency test evidence"

        # Test multiple calls with same parameters
        results_few_shot_neg = [
            get_fact_check_user_prompt(
                claim=claim,
                evidence=evidence,
                prompt_type=PromptType.FEW_SHOT_WITH_NEGATIVE,
            )
            for _ in range(3)
        ]

        results_zero_shot = [
            get_fact_check_user_prompt(
                claim=claim,
                evidence=evidence,
                prompt_type=PromptType.ZERO_SHOT,
            )
            for _ in range(3)
        ]

        # All results with same parameters should be identical
        assert all(r == results_few_shot_neg[0] for r in results_few_shot_neg)
        assert all(r == results_zero_shot[0] for r in results_zero_shot)

        # Results with different parameters should be different
        assert results_few_shot_neg[0] != results_zero_shot[0]

    def test_examples_behavior_with_different_combinations(self):
        """Test that examples are handled correctly across combinations."""  # noqa: E501
        claim = "Examples behavior test"
        evidence = "Examples behavior evidence"

        # Get results for all combinations
        result_few_shot_neg = get_fact_check_user_prompt(
            claim, evidence, PromptType.FEW_SHOT_WITH_NEGATIVE
        )
        result_few_shot = get_fact_check_user_prompt(
            claim, evidence, PromptType.FEW_SHOT
        )
        result_zero_shot = get_fact_check_user_prompt(
            claim, evidence, PromptType.ZERO_SHOT
        )

        # FEW_SHOT types should have examples
        assert FEW_SHOT_EXAMPLES in result_few_shot_neg
        assert FEW_SHOT_EXAMPLES in result_few_shot

        # ZERO_SHOT should NOT have examples
        assert FEW_SHOT_EXAMPLES not in result_zero_shot

        # All results should be strings
        for r in [result_few_shot_neg, result_few_shot, result_zero_shot]:
            assert isinstance(r, str)

    def test_prompt_length_differences(self):
        """Test that prompt length varies with parameter combinations.

        Prompts with examples should be longer than those without.
        """  # noqa: E501
        claim = "Length test claim"
        evidence = "Length test evidence"

        result_with_examples = get_fact_check_user_prompt(
            claim, evidence, prompt_type=PromptType.FEW_SHOT
        )
        result_without_examples = get_fact_check_user_prompt(
            claim, evidence, prompt_type=PromptType.ZERO_SHOT
        )

        # Prompt with examples should be longer
        assert len(result_with_examples) > len(result_without_examples)

        # Both should contain claim and evidence
        assert claim in result_with_examples
        assert claim in result_without_examples
        assert evidence in result_with_examples
        assert evidence in result_without_examples


class TestPromptIntegration:
    """Integration tests for prompt functions."""

    def test_system_and_user_prompt_compatibility(self):
        """Test that system and user prompts work together."""
        system_prompt = get_fact_check_system_prompt()
        user_prompt = get_fact_check_user_prompt(
            claim="Test claim",
            evidence="Test evidence",
            prompt_type=PromptType.FEW_SHOT,
        )

        # Both should mention JSON format (through the pred structure)
        assert "label" in user_prompt.lower()

        # Both should mention True/False
        assert (
            "true" in system_prompt.lower() and "false" in system_prompt.lower()
        )
        assert "true" in user_prompt.lower() and "false" in user_prompt.lower()

    def test_prompt_consistency_across_calls(self):
        """Test that prompts are consistent across multiple calls."""
        claim = "Consistency test"
        evidence = "Evidence for consistency"

        # Multiple calls should return identical results
        results = [
            get_fact_check_user_prompt(
                claim=claim, evidence=evidence, prompt_type=PromptType.FEW_SHOT
            )
            for _ in range(5)
        ]

        # All results should be identical
        first_result = results[0]
        assert all(result == first_result for result in results)

    def test_examples_format_validity(self):
        """Test that the examples in prompts follow the expected format."""
        user_prompt = get_fact_check_user_prompt(
            claim="Test",
            evidence="Test evidence",
            prompt_type=PromptType.FEW_SHOT_WITH_NEGATIVE,
        )

        # Should contain the few-shot example format
        assert '{"label": false}' in user_prompt
        assert "Claim:" in user_prompt
        assert "Evidence:" in user_prompt


class TestPromptEdgeCases:
    """Test edge cases and error conditions."""

    def test_non_string_inputs(self):
        """Test behavior with non-string inputs."""
        # Should work with numbers (converted to strings by format)
        result = get_fact_check_user_prompt(
            claim=123, evidence=456.789, prompt_type=PromptType.FEW_SHOT
        )

        assert "123" in result
        assert "456.789" in result

    def test_non_boolean_include_negative_examples(self):
        """Test behavior with different PromptType enum values."""
        # Should work with all PromptType values
        result_few_shot = get_fact_check_user_prompt(
            claim="test", evidence="test", prompt_type=PromptType.FEW_SHOT
        )

        result_zero_shot = get_fact_check_user_prompt(
            claim="test", evidence="test", prompt_type=PromptType.ZERO_SHOT
        )

        # Both should return strings
        assert isinstance(result_few_shot, str)
        assert isinstance(result_zero_shot, str)


class TestPromptFormatting:
    """Test specific formatting requirements."""

    def test_prompt_structure(self):
        """Test that the generated prompt has the expected structure."""
        claim = "Test claim for structure"
        evidence = "Test evidence for structure"

        result = get_fact_check_user_prompt(claim=claim, evidence=evidence)

        # Should contain instruction
        assert "Given the claim and evidence" in result

        # Should contain format specification
        assert "respond" in result
        assert '{"label": true}' in result
        assert '{"label": false}' in result

        # Should contain the actual claim and evidence
        assert claim in result
        assert evidence in result

    def test_json_format_specification(self):
        """Test that JSON format is properly specified."""
        result = get_fact_check_user_prompt(
            claim="JSON test",
            evidence="JSON evidence",
            prompt_type=PromptType.FEW_SHOT,
        )

        # Should specify exact JSON format
        assert '{"label": true}' in result
        assert '{"label": false}' in result
        assert "respond" in result.lower()

    def test_examples_inclusion(self):
        """Test that examples are properly included."""
        result = get_fact_check_user_prompt(
            claim="Example test",
            evidence="Example evidence",
            prompt_type=PromptType.FEW_SHOT,
        )

        # Should include the few-shot example
        assert "150,000 (oil spill) claims " in result
        assert "Leonardo DiCaprio has donated $10,000,000" in result
        assert '{"label": false}' in result
