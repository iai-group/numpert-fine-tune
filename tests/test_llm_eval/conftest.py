"""Shared pytest fixtures for llm_eval tests."""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock


@pytest.fixture(scope="session")
def base_config():
    """Base configuration dictionary for tests."""
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
            "num_ctx": 4096,
        },
    }


@pytest.fixture
def temp_config_file(base_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(base_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client with common methods."""
    client = Mock()
    client.pull = Mock()
    client.show = Mock()
    client.chat = Mock()
    return client


@pytest.fixture
def sample_messages():
    """Sample messages for testing chat functionality."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Generate 5 questions about climate change.",
        },
    ]


@pytest.fixture
def fact_check_messages():
    """Fact-checking specific messages for testing."""
    return [
        {
            "role": "system",
            "content": (
                "You are a professional fact checker. "
                "Generate questions to validate claims."
            ),
        },
        {
            "role": "user",
            "content": (
                "In a letter to Steve Jobs, Sean Connery refused to appear "
                "in an apple commercial."
            ),
        },
    ]
