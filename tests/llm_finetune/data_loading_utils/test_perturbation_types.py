import pytest

"""Tests for perturbation_types.py module."""


from src.llm_finetune.data_loading_utils.perturbation_types import (
    ALL_PERTURB_TYPES,
    OBFUSCATION_TYPES,
    STRUCTURE_PRESERVING_TYPES,
    VALUE_MODIFYING_TYPES,
    FileLoadConfig,
    PerturbationType,
    parse_file_config,
)


class TestPerturbationType:
    """Test PerturbationType enum."""

    def test_enum_values(self) -> None:
        """Test that all enum values are correctly defined."""
        assert (
            PerturbationType.ORIGINAL_NORMALIZED.value == "original_normalized"
        )
        assert PerturbationType.PERTURB_TYPE1.value == "perturb_type1"
        assert PerturbationType.PERTURB_TYPE2.value == "perturb_type2"
        assert (
            PerturbationType.PERTURB_TYPE3_RANDOM.value
            == "perturb_type3_random"
        )
        assert (
            PerturbationType.PERTURB_TYPE3_MASKED.value
            == "perturb_type3_masked"
        )
        assert (
            PerturbationType.PERTURB_TYPE3_REMOVED.value
            == "perturb_type3_removed"
        )

    def test_enum_members_count(self) -> None:
        """Test the total number of enum members."""
        assert len(PerturbationType) == 6


class TestConvenienceSets:
    """Test convenience sets for perturbation types."""

    def test_structure_preserving_types(self) -> None:
        """Test STRUCTURE_PRESERVING_TYPES set."""
        assert STRUCTURE_PRESERVING_TYPES == {PerturbationType.PERTURB_TYPE1}

    def test_value_modifying_types(self) -> None:
        """Test VALUE_MODIFYING_TYPES set."""
        assert VALUE_MODIFYING_TYPES == {PerturbationType.PERTURB_TYPE2}

    def test_obfuscation_types(self) -> None:
        """Test OBFUSCATION_TYPES set."""
        assert OBFUSCATION_TYPES == {
            PerturbationType.PERTURB_TYPE3_RANDOM,
            PerturbationType.PERTURB_TYPE3_MASKED,
            PerturbationType.PERTURB_TYPE3_REMOVED,
        }

    def test_all_perturb_types(self) -> None:
        """Test ALL_PERTURB_TYPES contains all non-original types."""
        expected = (
            STRUCTURE_PRESERVING_TYPES
            | VALUE_MODIFYING_TYPES
            | OBFUSCATION_TYPES
        )
        assert ALL_PERTURB_TYPES == expected
        assert len(ALL_PERTURB_TYPES) == 5

    def test_all_perturb_types_excludes_original_normalized(self) -> None:
        """Test that ORIGINAL_NORMALIZED is not in ALL_PERTURB_TYPES."""
        assert PerturbationType.ORIGINAL_NORMALIZED not in ALL_PERTURB_TYPES


class TestFileLoadConfig:
    """Test FileLoadConfig dataclass."""

    def test_init_with_path_only(self) -> None:
        """Test initialization with only path."""
        config = FileLoadConfig(path="test.json")
        assert config.path == "test.json"
        assert config.perturb_types is None

    def test_init_with_path_and_types(self) -> None:
        """Test initialization with path and perturbation types."""
        types = [PerturbationType.PERTURB_TYPE1, PerturbationType.PERTURB_TYPE2]
        config = FileLoadConfig(path="test.json", perturb_types=types)
        assert config.path == "test.json"
        assert config.perturb_types == types

    def test_repr_with_none_types(self) -> None:
        """Test __repr__ when perturb_types is None."""
        config = FileLoadConfig(path="test.json")
        assert (
            repr(config)
            == "FileLoadConfig(path='test.json', perturb_types=all)"
        )

    def test_repr_with_types(self) -> None:
        """Test __repr__ with specific perturbation types."""
        types = [PerturbationType.PERTURB_TYPE1, PerturbationType.PERTURB_TYPE2]
        config = FileLoadConfig(path="test.json", perturb_types=types)
        expected = (
            "FileLoadConfig(path='test.json', "
            "perturb_types=['perturb_type1', 'perturb_type2'])"
        )
        assert repr(config) == expected


class TestParseFileConfig:
    """Test parse_file_config function."""

    def test_parse_simple_path(self) -> None:
        """Test parsing a simple path without types."""
        config = parse_file_config("approx_True.json")
        assert config.path == "approx_True.json"
        assert config.perturb_types is None

    def test_parse_path_with_single_type(self) -> None:
        """Test parsing path with single perturbation type."""
        config = parse_file_config("approx_True.json:type1")
        assert config.path == "approx_True.json"
        assert config.perturb_types == [PerturbationType.PERTURB_TYPE1]

    def test_parse_path_with_multiple_types(self) -> None:
        """Test parsing path with multiple perturbation types."""
        config = parse_file_config("approx_True.json:type1,type2")
        assert config.path == "approx_True.json"
        assert config.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_parse_path_with_type3_variants(self) -> None:
        """Test parsing path with type3 variants."""
        config = parse_file_config(
            "obfusc_False.json:type3_random,type3_masked"
        )
        assert config.path == "obfusc_False.json"
        assert config.perturb_types == [
            PerturbationType.PERTURB_TYPE3_RANDOM,
            PerturbationType.PERTURB_TYPE3_MASKED,
        ]

    def test_parse_path_with_full_type_names(self) -> None:
        """Test parsing with full perturbation type names."""
        config = parse_file_config("test.json:perturb_type1,perturb_type2")
        assert config.path == "test.json"
        assert config.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_parse_path_with_whitespace_in_types(self) -> None:
        """Test parsing with whitespace around type names."""
        config = parse_file_config("test.json: type1 , type2 ")
        assert config.path == "test.json"
        assert config.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_parse_path_with_empty_type_list(self) -> None:
        """Test parsing with colon but no types specified."""
        config = parse_file_config("test.json:")
        assert config.path == "test.json"
        assert config.perturb_types is None

    def test_parse_path_with_colon_in_filename(self) -> None:
        """Test parsing path with colon using rsplit."""
        config = parse_file_config("path/to:file.json:type1")
        assert config.path == "path/to:file.json"
        assert config.perturb_types == [PerturbationType.PERTURB_TYPE1]

    def test_parse_invalid_type_raises_error(self) -> None:
        """Test that invalid perturbation type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_file_config("test.json:invalid_type")
        assert "Unknown perturbation type: 'invalid_type'" in str(
            exc_info.value
        )

    def test_parse_invalid_type_error_message(self) -> None:
        """Test error message lists valid types."""
        with pytest.raises(ValueError) as exc_info:
            parse_file_config("test.json:bad")
        error_msg = str(exc_info.value)
        assert "Valid types:" in error_msg
        assert "perturb_type1" in error_msg
        assert "perturb_type2" in error_msg

    def test_parse_mixed_valid_and_invalid_types(self) -> None:
        """Test that error is raised when one type is invalid."""
        with pytest.raises(ValueError):
            parse_file_config("test.json:type1,invalid")

    def test_parse_all_obfuscation_types(self) -> None:
        """Test parsing all obfuscation type variants."""
        config = parse_file_config(
            "obfusc.json:type3_random,type3_masked,type3_removed"
        )
        assert config.perturb_types == [
            PerturbationType.PERTURB_TYPE3_RANDOM,
            PerturbationType.PERTURB_TYPE3_MASKED,
            PerturbationType.PERTURB_TYPE3_REMOVED,
        ]
