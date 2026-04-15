"""Tests for GGUF quantization conversion module.

Tests cover:
- `save_as_gguf()` validation logic and call delegation
- `convert_existing_model()` default output path and plumbing
- CLI argument parsing
- Module-level constants
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from src.llm_finetune.utils.training.convert_to_gguf import (
    save_as_gguf,
    convert_existing_model,
    DEFAULT_QUANT_METHOD,
    SUPPORTED_QUANT_METHODS,
)


# Fixture to mock check_llamacpp_setup
@pytest.fixture(autouse=True)
def mock_llamacpp_check():
    """Mock check_llamacpp_setup to avoid requiring llama.cpp in tests."""
    with patch(
        "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
    ):
        yield


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are well-formed."""

    def test_default_quant_method_is_q4_k_m(self):
        assert DEFAULT_QUANT_METHOD == "q4_k_m"

    def test_default_in_supported(self):
        assert DEFAULT_QUANT_METHOD in SUPPORTED_QUANT_METHODS

    def test_supported_methods_non_empty(self):
        assert len(SUPPORTED_QUANT_METHODS) > 0

    def test_supported_methods_contains_expected(self):
        expected = {"q4_k_m", "q8_0", "f16", "q2_k"}
        assert expected.issubset(set(SUPPORTED_QUANT_METHODS))


# ---------------------------------------------------------------------------
# save_as_gguf
# ---------------------------------------------------------------------------


class TestSaveAsGguf:
    """Test the save_as_gguf helper function."""

    def _make_mocks(self):
        model = MagicMock()
        tokenizer = MagicMock()
        return model, tokenizer

    def test_rejects_unsupported_method(self, tmp_path):
        model, tokenizer = self._make_mocks()
        with pytest.raises(ValueError, match="Unsupported quantization"):
            save_as_gguf(
                model,
                tokenizer,
                output_dir=str(tmp_path / "out"),
                quantization_method="invalid_method",
            )

    def test_creates_output_dir(self, tmp_path):
        model, tokenizer = self._make_mocks()
        out = str(tmp_path / "nested" / "gguf_out")

        save_as_gguf(model, tokenizer, output_dir=out)

        assert os.path.isdir(out)

    def test_calls_save_pretrained_gguf_with_defaults(self, tmp_path):
        model, tokenizer = self._make_mocks()
        out = str(tmp_path / "out")

        save_as_gguf(model, tokenizer, output_dir=out)

        model.save_pretrained_gguf.assert_called_once_with(
            out,
            tokenizer,
            quantization_method=DEFAULT_QUANT_METHOD,
        )

    def test_passes_custom_method(self, tmp_path):
        model, tokenizer = self._make_mocks()
        out = str(tmp_path / "out")

        save_as_gguf(
            model, tokenizer, output_dir=out, quantization_method="q8_0"
        )

        model.save_pretrained_gguf.assert_called_once_with(
            out,
            tokenizer,
            quantization_method="q8_0",
        )

    def test_returns_output_dir(self, tmp_path):
        model, tokenizer = self._make_mocks()
        out = str(tmp_path / "out")

        result = save_as_gguf(model, tokenizer, output_dir=out)

        assert result == out

    @pytest.mark.parametrize("method", SUPPORTED_QUANT_METHODS)
    def test_all_supported_methods_accepted(self, tmp_path, method):
        """Every listed method should be accepted without raising."""
        model, tokenizer = self._make_mocks()
        out = str(tmp_path / method)

        save_as_gguf(
            model, tokenizer, output_dir=out, quantization_method=method
        )

        model.save_pretrained_gguf.assert_called_once()


# ---------------------------------------------------------------------------
# convert_existing_model
# ---------------------------------------------------------------------------


class TestConvertExistingModel:
    """Test the convert_existing_model orchestration function."""

    @patch("src.llm_finetune.utils.training.convert_to_gguf.subprocess.run")
    @patch("src.llm_finetune.utils.training.convert_to_gguf.os.makedirs")
    def test_default_output_dir(self, mock_makedirs, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        with patch(
            "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
        ) as mock_check:
            mock_check.return_value = MagicMock()
            mock_check.return_value.__truediv__ = MagicMock(
                side_effect=lambda x: MagicMock(
                    __str__=lambda s: f"/fake/llama.cpp/{x}",
                    exists=lambda s: True,
                )
            )
            convert_existing_model(model_path="/models/my_model")

        expected_out = "/models/my_model_gguf_q4_k_m"
        mock_makedirs.assert_called_once_with(expected_out, exist_ok=True)

    @patch("src.llm_finetune.utils.training.convert_to_gguf.subprocess.run")
    @patch("src.llm_finetune.utils.training.convert_to_gguf.os.makedirs")
    def test_custom_output_dir(self, mock_makedirs, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        with patch(
            "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
        ) as mock_check:
            mock_check.return_value = MagicMock()
            mock_check.return_value.__truediv__ = MagicMock(
                side_effect=lambda x: MagicMock(
                    __str__=lambda s: f"/fake/llama.cpp/{x}",
                    exists=lambda s: True,
                )
            )
            convert_existing_model(
                model_path="/models/my_model",
                output_dir="/custom/out",
            )

        mock_makedirs.assert_called_once_with("/custom/out", exist_ok=True)

    @patch("src.llm_finetune.utils.training.convert_to_gguf.subprocess.run")
    @patch("src.llm_finetune.utils.training.convert_to_gguf.os.makedirs")
    def test_rejects_unsupported_method(self, mock_makedirs, mock_run):
        with patch(
            "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
        ) as mock_check:
            mock_check.return_value = MagicMock()
            with pytest.raises(ValueError, match="Unsupported quantization"):
                convert_existing_model(
                    model_path="/models/m",
                    quantization_method="invalid_method",
                )

    @patch("src.llm_finetune.utils.training.convert_to_gguf.subprocess.run")
    @patch("src.llm_finetune.utils.training.convert_to_gguf.os.makedirs")
    def test_conversion_failure_raises(self, mock_makedirs, mock_run):
        mock_run.return_value = MagicMock(returncode=1)

        with patch(
            "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
        ) as mock_check:
            mock_check.return_value = MagicMock()
            mock_check.return_value.__truediv__ = MagicMock(
                side_effect=lambda x: MagicMock(
                    __str__=lambda s: f"/fake/llama.cpp/{x}",
                    exists=lambda s: True,
                )
            )
            with pytest.raises(RuntimeError, match="GGUF conversion failed"):
                convert_existing_model(model_path="/models/m")

    @patch("src.llm_finetune.utils.training.convert_to_gguf.subprocess.run")
    @patch("src.llm_finetune.utils.training.convert_to_gguf.os.makedirs")
    def test_trailing_slash_stripped_from_default_dir(
        self, mock_makedirs, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)

        with patch(
            "src.llm_finetune.utils.training.convert_to_gguf.check_llamacpp_setup"
        ) as mock_check:
            mock_check.return_value = MagicMock()
            mock_check.return_value.__truediv__ = MagicMock(
                side_effect=lambda x: MagicMock(
                    __str__=lambda s: f"/fake/llama.cpp/{x}",
                    exists=lambda s: True,
                )
            )
            convert_existing_model(model_path="/models/my_model/")

        expected_out = "/models/my_model_gguf_q4_k_m"
        mock_makedirs.assert_called_once_with(expected_out, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the CLI argparse configuration (without running conversion)."""

    def _parse_args(self, argv):
        """Parse CLI arguments using the module's argparse setup."""
        import argparse as _argparse

        parser = _argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default=None)
        parser.add_argument(
            "--quantization_method",
            type=str,
            default=DEFAULT_QUANT_METHOD,
            choices=SUPPORTED_QUANT_METHODS,
        )
        parser.add_argument("--max_seq_length", type=int, default=16384)
        return parser.parse_args(argv)

    def test_minimal_args(self):
        """--model_path is the only required argument."""
        args = self._parse_args(["--model_path", "/tmp/model"])

        assert args.model_path == "/tmp/model"
        assert args.output_dir is None
        assert args.quantization_method == DEFAULT_QUANT_METHOD
        assert args.max_seq_length == 16384

    def test_all_args(self):
        args = self._parse_args(
            [
                "--model_path",
                "/tmp/model",
                "--output_dir",
                "/tmp/out",
                "--quantization_method",
                "q8_0",
                "--max_seq_length",
                "8192",
            ]
        )

        assert args.model_path == "/tmp/model"
        assert args.output_dir == "/tmp/out"
        assert args.quantization_method == "q8_0"
        assert args.max_seq_length == 8192

    def test_invalid_quant_method_rejected(self):
        """Argparse should reject methods not in SUPPORTED_QUANT_METHODS."""
        with pytest.raises(SystemExit):
            self._parse_args(
                [
                    "--model_path",
                    "/tmp/model",
                    "--quantization_method",
                    "invalid",
                ]
            )
