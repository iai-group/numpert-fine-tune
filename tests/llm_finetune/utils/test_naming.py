"""Tests for src.llm_finetune.utils.naming module."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from src.llm_finetune.utils.naming import (
    ADJECTIVES,
    NOUNS,
    VERBS,
    build_output_dir_name,
    generate_run_name,
)


class TestGenerateRunName:
    """Test run name generation."""

    def test_basic_name_format(self):
        """Test that basic name includes model and word components."""
        name = generate_run_name("Qwen3-8B")
        parts = name.split("-")
        # Should have: Qwen3, 8B (from model), adjective, noun, verb
        assert len(parts) >= 4
        assert parts[0] == "Qwen3"
        assert parts[1] == "8B"
        # Adjective, noun, verb should be from our word lists
        assert parts[2] in ADJECTIVES
        assert parts[3] in NOUNS
        # Verb is the last part (has 's' at the end)
        assert parts[4].rstrip("s") in [v.rstrip("s") for v in VERBS]

    def test_model_name_preserved(self):
        """Test that model name is preserved in output."""
        name = generate_run_name("MyModel-16B")
        assert name.startswith("MyModel-16B-")

    def test_with_perturb_type_names(self):
        """Test that perturbation type is appended correctly."""
        name = generate_run_name("Qwen3-8B", perturb_type_names="numeration")
        assert "_numeration" in name
        assert name.startswith("Qwen3-8B-")

    def test_empty_perturb_type_names_omitted(self):
        """Test that empty perturbation names are not included."""
        name = generate_run_name("Qwen3-8B", perturb_type_names="")
        assert "_" not in name.split("-")[-1]  # No underscore at end

    def test_timestamp_parameter_optional(self):
        """Test that timestamp parameter is optional and accepted."""
        ts = datetime(2025, 3, 20, 14, 30, 22)
        # Should not raise an error even with timestamp (kept for compatibility)
        name = generate_run_name("model", timestamp=ts)
        assert name.startswith("model-")

    def test_uniqueness_without_directory(self):
        """Test that different calls produce different names."""
        names = {generate_run_name("model") for _ in range(10)}
        # Very unlikely to get same name twice (89000+ combinations)
        # but set to be safe
        assert len(names) >= 9  # At least 9 different names out of 10

    def test_uniqueness_with_existing_directory(self):
        """Test that function generates unique names avoiding existing dirs."""
        with TemporaryDirectory() as tmpdir:
            # Create a few directories with generated names
            first_name = generate_run_name("model", base_path=tmpdir)
            Path(tmpdir, first_name).mkdir()

            # Generate a second name; it should be different
            second_name = generate_run_name("model", base_path=tmpdir)
            assert first_name != second_name
            assert not Path(tmpdir, second_name).exists()

    def test_numeric_suffix_when_exhausted(self):
        """Test that numeric suffix is added when all combinations exhausted."""
        with TemporaryDirectory() as tmpdir:
            # Create a directory with a predictable name pattern
            existing_name = "model-ancient-ape-arranges"
            Path(tmpdir, existing_name).mkdir()

            # Mock the random choices to always pick the same words
            # to force collisions and exhaust max_attempts
            import src.llm_finetune.utils.naming as naming_module

            original_choice = naming_module.random.choice

            def mock_choice(seq):
                """Return same first element after attempting once."""
                if seq == naming_module.ADJECTIVES:
                    return "ancient"
                elif seq == naming_module.NOUNS:
                    return "ape"
                elif seq == naming_module.VERBS:
                    return "arranges"
                return original_choice(seq)

            # Patch random.choice
            naming_module.random.choice = mock_choice

            try:
                # With max_attempts=1, the first (and only) attempt will
                # collide with the existing directory
                name = generate_run_name(
                    "model", base_path=tmpdir, max_attempts=1
                )
                # Should have numeric suffix
                assert "_v" in name
            finally:
                # Restore original
                naming_module.random.choice = original_choice


class TestBuildOutputDirName:
    """Test output directory name construction."""

    def test_default_rank_16_no_suffix(self):
        result = build_output_dir_name("Qwen3-8B", rank=16, start_epoch=1)
        assert result == "Qwen3-8B-epoch-1"

    def test_rank_32_has_suffix(self):
        result = build_output_dir_name("Qwen3-8B", rank=32, start_epoch=1)
        assert result == "Qwen3-8B_r32-epoch-1"

    def test_unquantized(self):
        result = build_output_dir_name(
            "Qwen3-8B", rank=32, start_epoch=1, load_in_4bit=False
        )
        assert result == "Qwen3-8B_r32-unquantized-epoch-1"

    def test_epoch_0(self):
        result = build_output_dir_name("model", rank=32, start_epoch=0)
        assert result == "model_r32-epoch-0"
