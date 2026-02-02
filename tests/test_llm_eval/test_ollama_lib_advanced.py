"""Additional tests for ollama_lib.py.

This module focuses on thinking functionality and edge cases.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from ollama._types import ResponseError

from src.llm_eval.ollama_wrapper import OllamaLLM
from src.prompt import get_output_schema


class MockThinkingResponse:
    """Mock response with thinking capability."""

    def __init__(self, content: str, thinking: str = None):
        """Initialize mock thinking response.

        Args:
            content: The response content.
            thinking: Optional thinking content.
        """
        self.message = Mock()
        self.message.content = content
        self.message.thinking = thinking
        self.message.role = "assistant"
        self.message.images = None
        self.message.tool_calls = None
        self.model = "qwen3:8b"
        self.created_at = "2025-08-05T12:00:00Z"
        self.done = True


class TestOllamaLLMThinking:
    """Test cases specifically for thinking functionality."""

    def test_thinking_enabled_in_config(
        self, temp_config_file, mock_ollama_client
    ):
        """Test that thinking parameter is correctly set from config."""
        mock_response = MockThinkingResponse(
            content="Answer content", thinking="Step-by-step reasoning process"
        )
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            messages = [{"role": "user", "content": "Test thinking"}]

            with patch("builtins.print"):
                response = llm.chat(messages)

            # Verify thinking was enabled in the call
            call_args = mock_ollama_client.chat.call_args
            assert call_args[1]["think"] is True
            assert response.message.thinking == "Step-by-step reasoning process"

    def test_thinking_disabled_in_config(self, base_config, mock_ollama_client):
        """Test chat with thinking explicitly disabled."""
        # Create config with thinking disabled
        config_no_think = base_config.copy()
        config_no_think["think"] = False

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_no_think, f)
            config_path = f.name

        try:
            mock_response = MockThinkingResponse(
                content="Answer without thinking", thinking=None
            )
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                messages = [{"role": "user", "content": "Test"}]

                with patch("builtins.print"):
                    response = llm.chat(messages)

                # Verify thinking was disabled
                call_args = mock_ollama_client.chat.call_args
                assert call_args[1]["think"] is False
                assert response.message.thinking is None
        finally:
            os.unlink(config_path)

    def test_thinking_missing_from_config(
        self, base_config, mock_ollama_client
    ):
        """Test behavior when thinking is not specified in config."""
        # Remove thinking from config
        config_no_think_key = base_config.copy()
        del config_no_think_key["think"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_no_think_key, f)
            config_path = f.name

        try:
            mock_response = MockThinkingResponse(content="Default behavior")
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                # Should default to False
                assert llm._think is False

                messages = [{"role": "user", "content": "Test"}]

                with patch("builtins.print"):
                    _ = llm.chat(messages)

                call_args = mock_ollama_client.chat.call_args
                assert call_args[1]["think"] is False
        finally:
            os.unlink(config_path)


class TestOllamaLLMEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config_file(self, mock_ollama_client):
        """Test handling of empty config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")  # Empty file
            config_path = f.name

        try:
            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                # Should handle None values gracefully
                assert llm._model_name is None
                assert llm._model_options == {}
                assert llm._stream is False
                assert llm._think is False
        finally:
            os.unlink(config_path)

    def test_malformed_config_values(self, mock_ollama_client):
        """Test handling of malformed config values."""
        malformed_config = {
            "model": 123,  # Should be string
            "stream": "not_boolean",  # Should be boolean
            "think": "yes",  # Should be boolean
            "options": "not_dict",  # Should be dict
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(malformed_config, f)
            config_path = f.name

        try:
            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                # Should handle type mismatches gracefully
                assert llm._model_name == 123
                assert llm._stream == "not_boolean"
                assert llm._think == "yes"
                assert llm._model_options == "not_dict"
        finally:
            os.unlink(config_path)

    def test_chat_with_empty_messages(
        self, temp_config_file, mock_ollama_client
    ):
        """Test chat with empty message list."""
        mock_response = MockThinkingResponse(content="Empty response")
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            with patch("builtins.print"):
                response = llm.chat([])

            mock_ollama_client.chat.assert_called_once()
            assert response.message.content == "Empty response"
            assert response.message.content == "Empty response"

    def test_chat_with_none_messages(
        self, temp_config_file, mock_ollama_client
    ):
        """Test chat with None message content."""
        messages = [
            {"role": "user", "content": None},
            {"role": "system", "content": ""},
        ]

        mock_response = MockThinkingResponse(content="Handled none content")
        mock_ollama_client.chat.return_value = mock_response

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            with patch("builtins.print"):
                response = llm.chat(messages)
                assert response

            mock_ollama_client.chat.assert_called_once()

    def test_multiple_consecutive_errors(
        self, temp_config_file, mock_ollama_client
    ):
        """Test handling of multiple consecutive errors."""
        # Setup: first call fails, then pull succeeds, then retry succeeds
        mock_ollama_client.chat.side_effect = [
            ResponseError("model not found", 404),  # First attempt fails
            MockThinkingResponse(
                content="Success after pull"
            ),  # Retry succeeds
        ]

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            messages = [{"role": "user", "content": "Test"}]

            # First call should pull model and succeed on retry
            with (
                patch("builtins.print"),
                patch("code.llm_eval.ollama_wrapper.logging"),
            ):
                response1 = llm.chat(messages)

            assert response1.message.content == "Success after pull"
            # Verify that chat was called twice (initial + retry)
            assert mock_ollama_client.chat.call_count == 2
            # Verify that pull was called once
            mock_ollama_client.pull.assert_called_once_with("qwen3:8b")

        # Test case where retry also fails
        mock_ollama_client.reset_mock()
        mock_ollama_client.chat.side_effect = [
            ResponseError("model not found", 404),  # First attempt fails
            ResponseError("model still not found", 404),  # Retry also fails
        ]

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            messages = [{"role": "user", "content": "Test"}]

            # This should fail since both attempts fail
            with pytest.raises(ResponseError):
                with (
                    patch("builtins.print"),
                    patch("code.llm_eval.ollama_wrapper.logging"),
                ):
                    llm.chat(messages)

    def test_network_timeout_simulation(
        self, temp_config_file, mock_ollama_client
    ):
        """Test handling of network timeout scenarios."""
        import socket

        mock_ollama_client.chat.side_effect = socket.timeout("Network timeout")

        with patch(
            "code.llm_eval.ollama_wrapper.Client",
            return_value=mock_ollama_client,
        ):
            llm = OllamaLLM(config_path=temp_config_file)

            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(socket.timeout):
                with patch("builtins.print"):
                    llm.chat(messages)

    def test_model_pull_with_special_characters(
        self, temp_config_file, mock_ollama_client
    ):
        """Test model pulling with special characters in model name."""
        # Create config with special model name
        special_config = {"model": "test/model:v1.0-beta", "options": {}}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(special_config, f)
            config_path = f.name

        try:
            mock_ollama_client.chat.side_effect = ResponseError(
                "model not found", 404
            )

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                messages = [{"role": "user", "content": "Test"}]

                with pytest.raises(ResponseError):  # Will fail on retry
                    with (
                        patch("builtins.print"),
                        patch("code.llm_eval.ollama_wrapper.logging"),
                    ):
                        llm.chat(messages)

                # Verify pull was called with special model name
                mock_ollama_client.pull.assert_called_once_with(
                    "test/model:v1.0-beta"
                )
        finally:
            os.unlink(config_path)


class TestOllamaLLMConfiguration:
    """Test various configuration scenarios."""

    @pytest.mark.parametrize("format_type", ["json", "yaml", "xml", None])
    def test_different_format_configurations(
        self, base_config, mock_ollama_client, format_type
    ):
        """Test different format configurations."""
        config = base_config.copy()
        if format_type:
            config["format"] = format_type

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            mock_response = MockThinkingResponse(
                content=f"Response in {format_type}"
            )
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                messages = [{"role": "user", "content": "Test format"}]

                with patch("builtins.print"):
                    response = llm.chat(messages)
                    assert response

                call_args = mock_ollama_client.chat.call_args
                if not call_args[1]["think"]:
                    assert call_args[1]["format"] == get_output_schema
                else:
                    # Should not have format parameter if not specified
                    assert "format" not in call_args[1]
        finally:
            os.unlink(config_path)

    @pytest.mark.parametrize("stream_value", [True, False, None])
    def test_stream_configurations(
        self, base_config, mock_ollama_client, stream_value
    ):
        """Test different stream configurations."""
        config = base_config.copy()
        if stream_value is not None:
            config["stream"] = stream_value
        else:
            del config["stream"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            mock_response = MockThinkingResponse(content="Streaming test")
            mock_ollama_client.chat.return_value = mock_response

            with patch(
                "code.llm_eval.ollama_wrapper.Client",
                return_value=mock_ollama_client,
            ):
                llm = OllamaLLM(config_path=config_path)

                expected_stream = (
                    stream_value if stream_value is not None else False
                )
                assert llm._stream == expected_stream

                messages = [{"role": "user", "content": "Test streaming"}]

                with patch("builtins.print"):
                    response = llm.chat(messages)
                    assert response

                call_args = mock_ollama_client.chat.call_args
                assert call_args[1]["stream"] == expected_stream
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
