import pytest

"""Tests for perturbation_logic.py."""


from src.llm_finetune.data_loading_utils.perturbation_logic import (
    _detect_perturb_key,
    _determine_perturbed_label,
    _normalize_label,
    expand_perturbations,
)
from src.llm_finetune.data_loading_utils.perturbation_types import (
    PerturbationType,
)


class TestDetectPerturbKey:
    """Tests for _detect_perturb_key function."""

    def test_perturb_type1_true(self):
        """Should detect perturb_type1_True."""
        assert _detect_perturb_key("perturb_type1_True") == "perturb_type1"

    def test_perturb_type1_false(self):
        """Should detect perturb_type1_False."""
        assert _detect_perturb_key("perturb_type1_False") == "perturb_type1"

    def test_perturb_type1_no_suffix(self):
        """Should detect perturb_type1 without suffix."""
        assert _detect_perturb_key("perturb_type1") == "perturb_type1"

    def test_perturb_type2_true(self):
        """Should detect perturb_type2_True."""
        assert _detect_perturb_key("perturb_type2_True") == "perturb_type2"

    def test_perturb_type3_random_false(self):
        """Should detect perturb_type3_random_False."""
        assert (
            _detect_perturb_key("perturb_type3_random_False")
            == "perturb_type3_random"
        )

    def test_perturb_type3_masked(self):
        """Should detect perturb_type3_masked."""
        assert (
            _detect_perturb_key("perturb_type3_masked")
            == "perturb_type3_masked"
        )

    def test_perturb_type3_removed_true(self):
        """Should detect perturb_type3_removed_True."""
        assert (
            _detect_perturb_key("perturb_type3_removed_True")
            == "perturb_type3_removed"
        )

    def test_non_perturbation_key(self):
        """Should return None for non-perturbation keys."""
        assert _detect_perturb_key("claim_id") is None
        assert _detect_perturb_key("normalized_claim") is None
        assert _detect_perturb_key("doc") is None

    def test_invalid_perturbation_format(self):
        """Should return None for invalid perturbation formats."""
        assert _detect_perturb_key("perturb_type4_True") is None
        assert _detect_perturb_key("perturb_type1_Invalid") is None


class TestNormalizeLabel:
    """Tests for _normalize_label function."""

    def test_bool_true(self):
        """Should normalize True (bool) to 'True' (str)."""
        assert _normalize_label(True) == "True"

    def test_bool_false(self):
        """Should normalize False (bool) to 'False' (str)."""
        assert _normalize_label(False) == "False"

    def test_string_true_lowercase(self):
        """Should normalize 'true' (str) to 'True'."""
        assert _normalize_label("true") == "True"

    def test_string_true_uppercase(self):
        """Should normalize 'TRUE' (str) to 'True'."""
        assert _normalize_label("TRUE") == "True"

    def test_string_true_mixed_case(self):
        """Should normalize 'TrUe' (str) to 'True'."""
        assert _normalize_label("TrUe") == "True"

    def test_string_false_lowercase(self):
        """Should normalize 'false' (str) to 'False'."""
        assert _normalize_label("false") == "False"

    def test_string_false_uppercase(self):
        """Should normalize 'FALSE' (str) to 'False'."""
        assert _normalize_label("FALSE") == "False"

    def test_string_with_whitespace(self):
        """Should normalize strings with leading/trailing whitespace."""
        assert _normalize_label("  true  ") == "True"
        assert _normalize_label("\tfalse\n") == "False"

    def test_invalid_string(self):
        """Should raise ValueError for invalid string labels."""
        with pytest.raises(ValueError):
            _normalize_label("maybe")

    def test_invalid_type(self):
        """Should raise ValueError for invalid types."""
        with pytest.raises(ValueError):
            _normalize_label(1)
        with pytest.raises(ValueError):
            _normalize_label(None)


class TestDeterminePerturbedLabel:
    """Tests for _determine_perturbed_label function."""

    def test_perturb_type1_original_true(self):
        """Type1 (structure-preserving) keeps original True."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE1.value, True
        )
        assert label == "True"

    def test_perturb_type1_original_false(self):
        """Type1 (structure-preserving) keeps original False."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE1.value, False
        )
        assert label == "False"

    def test_perturb_type2_original_true(self):
        """Type2 (value-modifying) flips original True to False."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE2.value, True
        )
        assert label == "False"

    def test_perturb_type2_original_false(self):
        """Type2 (value-modifying) flips original False to True."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE2.value, False
        )
        assert label == "True"

    def test_perturb_type3_random_original_true(self):
        """Type3 (obfuscation) always returns False regardless of original."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE3_RANDOM.value, True
        )
        assert label == "False"

    def test_perturb_type3_random_original_false(self):
        """Type3 (obfuscation) always returns False regardless of original."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE3_RANDOM.value, False
        )
        assert label == "False"

    def test_perturb_type3_masked(self):
        """Type3 masked always returns False."""
        label = _determine_perturbed_label(
            PerturbationType.PERTURB_TYPE3_MASKED.value, True
        )
        assert label == "False"


class TestExpandPerturbations:
    """Tests for expand_perturbations function."""

    def test_single_example_original_only(self):
        """Should include original entry even without perturbations."""
        examples = [
            {
                "claim_id": "1",
                "normalized_claim": "The sky is blue.",
                "label": "True",
                "doc": "Evidence about sky color",
                "url": "http://example.com",
                "taxonomy_label": "weather",
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 1
        assert expanded[0]["perturbation_type"] == "original_normalized"
        assert expanded[0]["claim"] == "The sky is blue."
        assert expanded[0]["label"] == "True"

    def test_example_with_perturb_type1(self):
        """Should expand example with perturb_type1 perturbation."""
        examples = [
            {
                "claim_id": "1",
                "normalized_claim": "Original claim",
                "label": True,
                "doc": "Evidence",
                "perturb_type1": "Perturbed claim type1",
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 2
        assert expanded[0]["perturbation_type"] == "original_normalized"
        assert expanded[0]["label"] == "True"
        assert expanded[1]["perturbation_type"] == "perturb_type1"
        assert expanded[1]["claim"] == "Perturbed claim type1"
        assert expanded[1]["label"] == "True"

    def test_example_with_perturb_type2(self):
        """Should expand example with perturb_type2 perturbation."""
        examples = [
            {
                "claim_id": "2",
                "normalized_claim": "Original claim",
                "label": False,
                "doc": "Evidence",
                "perturb_type2": "Perturbed claim type2",
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 2
        assert expanded[1]["perturbation_type"] == "perturb_type2"
        assert expanded[1]["label"] == "True"  # Flipped from False

    def test_example_with_perturb_type3_random(self):
        """Should expand example with perturb_type3_random perturbation."""
        examples = [
            {
                "claim_id": "3",
                "normalized_claim": "Original claim",
                "label": True,
                "doc": "Evidence",
                "perturb_type3_random": "Perturbed claim type3",
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 2
        assert expanded[1]["perturbation_type"] == "perturb_type3_random"
        assert expanded[1]["label"] == "False"  # Always False for type3

    def test_multiple_perturbations_same_example(self):
        """Should expand example with multiple perturbation types."""
        examples = [
            {
                "claim_id": "4",
                "normalized_claim": "Original",
                "label": "true",
                "doc": "Evidence",
                "perturb_type1": "Perturbed 1",
                "perturb_type2": "Perturbed 2",
                "perturb_type3_masked": "Perturbed 3",
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 4
        assert expanded[0]["perturbation_type"] == "original_normalized"
        assert expanded[1]["perturbation_type"] == "perturb_type1"
        assert expanded[2]["perturbation_type"] == "perturb_type2"
        assert expanded[3]["perturbation_type"] == "perturb_type3_masked"

    def test_empty_perturbation_values_skipped(self):
        """Should skip perturbation entries with empty values."""
        examples = [
            {
                "claim_id": "5",
                "normalized_claim": "Original",
                "label": True,
                "doc": "Evidence",
                "perturb_type1": "",
                "perturb_type2": None,
            }
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 1
        assert expanded[0]["perturbation_type"] == "original_normalized"

    def test_allowed_types_filtering(self):
        """Should filter by allowed_types parameter."""
        examples = [
            {
                "claim_id": "6",
                "normalized_claim": "Original",
                "label": True,
                "doc": "Evidence",
                "perturb_type1": "Perturbed 1",
                "perturb_type2": "Perturbed 2",
                "perturb_type3_random": "Perturbed 3",
            }
        ]
        allowed = [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE3_RANDOM,
        ]
        expanded = expand_perturbations(examples, allowed_types=allowed)
        assert len(expanded) == 3  # Original + type1 + type3_random
        types = [e["perturbation_type"] for e in expanded]
        assert "perturb_type1" in types
        assert "perturb_type3_random" in types
        assert "perturb_type2" not in types

    def test_original_always_included_with_allowed_types(self):
        """Original should always be included even without explicit
        allowance."""
        examples = [
            {
                "claim_id": "7",
                "normalized_claim": "Original",
                "label": True,
                "doc": "Evidence",
                "perturb_type1": "Perturbed",
            }
        ]
        allowed = []
        expanded = expand_perturbations(examples, allowed_types=allowed)
        assert len(expanded) == 1
        assert expanded[0]["perturbation_type"] == "original_normalized"

    def test_preserves_optional_fields(self):
        """Should preserve optional url and taxonomy_label fields."""
        examples = [
            {
                "claim_id": "8",
                "normalized_claim": "Original",
                "label": True,
                "doc": "Evidence",
                "url": "http://example.com",
                "taxonomy_label": "politics",
            }
        ]
        expanded = expand_perturbations(examples)
        assert expanded[0]["url"] == "http://example.com"
        assert expanded[0]["taxonomy_label"] == "politics"

    def test_missing_optional_fields_defaults(self):
        """Should use empty string for missing optional fields."""
        examples = [
            {
                "claim_id": "9",
                "normalized_claim": "Original",
                "label": False,
                "doc": "Evidence",
            }
        ]
        expanded = expand_perturbations(examples)
        assert expanded[0]["url"] == ""
        assert expanded[0]["taxonomy_label"] == ""

    def test_preserves_claim_id_and_doc(self):
        """Should preserve claim_id and doc across all variants."""
        examples = [
            {
                "claim_id": "test_id_123",
                "normalized_claim": "Original",
                "label": True,
                "doc": "Test evidence document",
                "perturb_type1": "Perturbed",
            }
        ]
        expanded = expand_perturbations(examples)
        for entry in expanded:
            assert entry["claim_id"] == "test_id_123"
            assert entry["doc"] == "Test evidence document"

    def test_multiple_examples(self):
        """Should handle multiple examples correctly."""
        examples = [
            {
                "claim_id": "1",
                "normalized_claim": "Claim 1",
                "label": True,
                "doc": "Doc 1",
                "perturb_type1": "Perturbed 1a",
            },
            {
                "claim_id": "2",
                "normalized_claim": "Claim 2",
                "label": False,
                "doc": "Doc 2",
                "perturb_type2": "Perturbed 2a",
            },
        ]
        expanded = expand_perturbations(examples)
        assert len(expanded) == 4
        claim_ids = [e["claim_id"] for e in expanded]
        assert claim_ids.count("1") == 2
        assert claim_ids.count("2") == 2

    def test_label_normalization_in_expansion(self):
        """Should normalize labels during expansion."""
        examples = [
            {
                "claim_id": "10",
                "normalized_claim": "Original",
                "label": "TRUE",
                "doc": "Evidence",
            }
        ]
        expanded = expand_perturbations(examples)
        assert expanded[0]["label"] == "True"
