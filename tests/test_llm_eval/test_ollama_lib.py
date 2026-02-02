"""Test suite for ollama_lib.py module.

This module tests the OllamaLLM class with mocked Ollama server responses.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from ollama._types import ResponseError

from src.llm_eval.ollama_wrapper import OllamaLLM
from src.prompt import get_output_schema


class MockMessage:
    """Mock message object to simulate Ollama response message."""

    def __init__(
        self, content: str, thinking: str = None, role: str = "assistant"
    ):
        """Initialize mock message.

        Args:
            content: The message content.
            thinking: Optional thinking content.
            role: The role of the message sender.
        """
        self.content = content
        self.thinking = thinking
        self.role = role
        self.images = None
        self.tool_calls = None


class MockResponse:
    """Mock response object to simulate Ollama chat response."""

    def __init__(
        self, content: str, thinking: str = None, model: str = "test-model"
    ):
        """Initialize mock response.

        Args:
            content: The response content.
            thinking: Optional thinking content.
            model: The model name.
        """
        self.message = MockMessage(content, thinking)
        self.model = model
        self.created_at = "2025-08-05T12:00:00Z"
        self.done = True
        self.done_reason = "stop"
        self.total_duration = 1000000
        self.load_duration = 100000
        self.prompt_eval_count = 50
        self.prompt_eval_duration = 200000
        self.eval_count = 100
        self.eval_duration = 700000


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "qwen3:8b",
        "stream": False,
        "think": True,
        "options": {
            "seed": 42,
            "num_predict": 1000,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.7,
        },
    }


@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client with common methods."""
    client = Mock()
    client.pull = Mock()
    client.show = Mock()
    client.chat = Mock()
    return client


class TestOllamaLLM:
    """Test cases for OllamaLLM class."""

    def test_init_success(self, config_file, mock_ollama_client):
        """Test successful initialization of OllamaLLM."""
        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            assert llm._model_name == "qwen3:8b"
            assert llm._stream is False
            assert llm._think is True
            assert llm._model_options["seed"] == 42
            assert llm._client == mock_ollama_client

    def test_init_file_not_found(self):
        """Test initialization with non-existent config file."""
        with pytest.raises(FileNotFoundError):
            OllamaLLM(config_path="non_existent_file.yaml")

    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            invalid_config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                OllamaLLM(config_path=invalid_config_path)
        finally:
            os.unlink(invalid_config_path)

    def test_load_config_success(self, sample_config, config_file):
        """Test successful config loading."""
        config = OllamaLLM._load_config(config_file)
        assert config == sample_config

    def test_chat_success(self, config_file, mock_ollama_client):
        """Test successful chat completion."""
        # Setup mock response
        mock_response = MockResponse(
            content=(
                '{"questions": [{"question": "Test question?", '
                '"answer_type": "Boolean"}]}'
            ),
            thinking="Let me think about this step by step...",
        )
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test question"},
            ]

            with patch("builtins.print"):  # Suppress debug prints
                response = llm.chat(messages)

            assert response == mock_response
            assert response.message.content.startswith('{"questions":')
            assert (
                response.message.thinking
                == "Let me think about this step by step..."
            )

            # Verify that chat was called with correct parameters
            mock_ollama_client.chat.assert_called_once()
            call_args = mock_ollama_client.chat.call_args
            assert call_args[1]["model"] == "qwen3:8b"
            assert call_args[1]["stream"] is False
            assert call_args[1]["think"] is True

    def test_chat_with_json_format_detection(
        self, config_file, mock_ollama_client
    ):
        """Test chat with automatic JSON format detection."""
        mock_response = MockResponse(content='{"result": "test"}')
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            messages = [
                {"role": "user", "content": "Return result in JSON format"}
            ]

            with patch("builtins.print"):
                _ = llm.chat(messages)

            # Verify JSON format was detected and added
            call_args = mock_ollama_client.chat.call_args
            if not call_args[1]["think"]:
                assert call_args[1]["format"] == get_output_schema()

    def test_chat_model_not_found(self, config_file, mock_ollama_client):
        """Test chat when model is not found and needs to be pulled."""
        # First call raises 404 error
        mock_ollama_client.chat.side_effect = [
            ResponseError("model not found", 404),
            MockResponse(content="Success after pull"),
        ]

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            messages = [{"role": "user", "content": "Test"}]

            with patch("builtins.print"):
                response = llm.chat(messages)

            # Verify model was pulled and chat was retried
            mock_ollama_client.pull.assert_called_once_with("qwen3:8b")
            assert mock_ollama_client.chat.call_count == 2
            assert response.message.content == "Success after pull"

    def test_chat_other_error(self, config_file, mock_ollama_client):
        """Test chat with non-404 error."""
        mock_ollama_client.chat.side_effect = ResponseError("Server error", 500)

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ResponseError) as exc_info:
                with patch("builtins.print"):
                    llm.chat(messages)

            assert exc_info.value.status_code == 500
            # Verify pull was not called for non-404 errors
            mock_ollama_client.pull.assert_not_called()

    def test_pull_model_success(self, config_file, mock_ollama_client):
        """Test successful model pulling."""
        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            with (
                patch("builtins.print"),
                patch("code.llm_eval.ollama_wrapper.logging"),
            ):
                llm._pull_model("test-model")

            mock_ollama_client.pull.assert_called_once_with("test-model")

    def test_pull_model_failure(self, config_file, mock_ollama_client):
        """Test model pulling failure."""
        mock_ollama_client.pull.side_effect = Exception("Network error")

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            with pytest.raises(Exception) as exc_info:
                with (
                    patch("builtins.print"),
                    patch("code.llm_eval.ollama_wrapper.logging"),
                ):
                    llm._pull_model("test-model")

            assert "Failed to pull model 'test-model'" in str(exc_info.value)

    def test_check_model_capabilities_success(
        self, config_file, mock_ollama_client
    ):
        """Test successful model capabilities check."""
        mock_info = {
            "details": {"family": "qwen", "format": "gguf", "parameters": "8B"}
        }
        mock_ollama_client.show.return_value = mock_info

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            capabilities = llm.check_model_capabilities()

            assert capabilities == mock_info
            mock_ollama_client.show.assert_called_once_with("qwen3:8b")

    def test_check_model_capabilities_failure(
        self, config_file, mock_ollama_client
    ):
        """Test model capabilities check failure."""
        mock_ollama_client.show.side_effect = Exception("Model not available")

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            with patch("code.llm_eval.ollama_wrapper.logging"):
                capabilities = llm.check_model_capabilities()

            assert capabilities == {}

    def test_chat_without_thinking(self, config_file, mock_ollama_client):
        """Test chat with thinking disabled."""
        # Create config without thinking
        config_no_think = {
            "model": "qwen3:8b",
            "stream": False,
            "think": False,
            "options": {"seed": 42},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_no_think, f)
            config_path = f.name

        try:
            mock_response = MockResponse(content="Response without thinking")
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                messages = [{"role": "user", "content": "Test"}]

                with patch("builtins.print"):
                    _ = llm.chat(messages)

                # Verify thinking was not enabled
                call_args = mock_ollama_client.chat.call_args
                assert call_args[1]["think"] is False
        finally:
            os.unlink(config_path)

    def test_chat_with_custom_format(self, config_file, mock_ollama_client):
        """Test chat with custom format specified in config."""
        # Create config with custom format
        config_with_format = {
            "model": "qwen3:8b",
            "format": "yaml",
            "options": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_with_format, f)
            config_path = f.name

        try:
            mock_response = MockResponse(content="yaml: content")
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                messages = [{"role": "user", "content": "Test"}]

                with patch("builtins.print"):
                    _ = llm.chat(messages)

                # Verify custom format was used
                call_args = mock_ollama_client.chat.call_args
                if call_args[1]["think"]:
                    assert call_args[1]["format"] == get_output_schema()
        finally:
            os.unlink(config_path)


class TestOllamaLLMIntegration:
    """Integration tests for OllamaLLM."""

    @pytest.mark.parametrize("thinking_enabled", [True, False])
    def test_full_workflow(
        self, config_file, mock_ollama_client, thinking_enabled
    ):
        """Test complete workflow with different thinking configurations."""
        # Modify config for thinking parameter
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config["think"] = thinking_enabled

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Setup mock responses
        mock_info = {"details": {"family": "qwen", "format": "gguf"}}
        mock_ollama_client.show.return_value = mock_info

        thinking_content = "Step 1: Analyze..." if thinking_enabled else None
        mock_response = MockResponse(
            content='{"result": "success"}', thinking=thinking_content
        )
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            # Check capabilities
            capabilities = llm.check_model_capabilities()
            assert capabilities == mock_info

            # Perform chat
            messages = [{"role": "user", "content": "Test JSON response"}]

            with patch("builtins.print"):
                response = llm.chat(messages)

            assert response.message.content == '{"result": "success"}'
            if thinking_enabled:
                assert response.message.thinking == "Step 1: Analyze..."
            else:
                assert response.message.thinking is None

    def test_error_handling_chain(self, config_file, mock_ollama_client):
        """Test error handling in a chain of operations."""
        # Setup: model not found, then pull fails, then success
        mock_ollama_client.chat.side_effect = [
            ResponseError("model not found", 404),
            MockResponse(content="Success after retry"),
        ]

        # First pull fails, second succeeds
        mock_ollama_client.pull.side_effect = [
            Exception("Network error"),
            None,  # Success on retry
        ]

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=config_file)

            messages = [{"role": "user", "content": "Test"}]

            # First attempt should fail due to pull error
            with pytest.raises(Exception):
                with (
                    patch("builtins.print"),
                    patch("code.llm_eval.ollama_wrapper.logging"),
                ):
                    llm.chat(messages)

            # Reset for successful scenario
            mock_ollama_client.pull.side_effect = None
            mock_ollama_client.chat.side_effect = [
                ResponseError("model not found", 404),
                MockResponse(content="Success after pull"),
            ]

            with (
                patch("builtins.print"),
                patch("code.llm_eval.ollama_wrapper.logging"),
            ):
                response = llm.chat(messages)

            assert response.message.content == "Success after pull"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
