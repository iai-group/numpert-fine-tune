"""Unit tests for PEFT fine-tuning module."""

import pytest
from src.llm_finetune.peft_finetune import (
    PeftMethod,
    resolve_step_interval,
    get_model_name_arg,
    EfficiencyTracker,
    count_trainable_params,
)


class TestPeftMethod:
    """Test PeftMethod enum."""

    def test_peft_methods_exist(self):
        """Test that all PEFT methods are defined."""
        assert PeftMethod.LORA
        assert PeftMethod.PROMPT_TUNING
        assert PeftMethod.PREFIX_TUNING
        assert PeftMethod.LORA_PROMPT
        assert PeftMethod.LORA_PREFIX

    def test_peft_method_values(self):
        """Test PEFT method string values."""
        assert PeftMethod.LORA.value == "lora"
        assert PeftMethod.PROMPT_TUNING.value == "prompt_tuning"
        assert PeftMethod.PREFIX_TUNING.value == "prefix_tuning"
        assert PeftMethod.LORA_PROMPT.value == "lora_prompt"
        assert PeftMethod.LORA_PREFIX.value == "lora_prefix"

    def test_peft_method_from_string(self):
        """Test creating PeftMethod from string."""
        method = PeftMethod("lora")
        assert method == PeftMethod.LORA


class TestResolveStepInterval:
    """Test step interval resolution."""

    def test_absolute_steps(self):
        """Test absolute step values."""
        assert resolve_step_interval(100) == 100
        assert resolve_step_interval(500) == 500

    def test_fractional_steps(self):
        """Test fractional step values (0, 1]."""
        # 0.5 of 1000 = 500
        result = resolve_step_interval(0.5, total_steps=1000)
        assert result == 500

        # 0.25 of 400 = 100
        result = resolve_step_interval(0.25, total_steps=400)
        assert result == 100

    def test_fractional_requires_total_steps(self):
        """Test that fractional steps require total_steps."""
        with pytest.raises(ValueError, match="total_steps is required"):
            resolve_step_interval(0.5)

    def test_invalid_steps(self):
        """Test invalid step values."""
        with pytest.raises(ValueError, match="must be positive"):
            resolve_step_interval(-100)

        with pytest.raises(ValueError, match="must be positive"):
            resolve_step_interval(0)

        with pytest.raises(ValueError, match="must be positive"):
            resolve_step_interval(-1.5)


class TestGetModelNameArg:
    """Test model name argument conversion."""

    def test_qwen_3b_conversion(self):
        """Test Qwen 3.2 8B model conversion."""
        result = get_model_name_arg("Qwen3-8B")
        assert result == "Qwen/Qwen2.5-8B-Instruct"

    def test_qwen_32b_conversion(self):
        """Test Qwen 32B model conversion."""
        result = get_model_name_arg("Qwen3-32B")
        assert result == "Qwen/Qwen2.5-32B-Instruct"

    def test_llama_3b_conversion(self):
        """Test Llama 3.2 3B model conversion."""
        result = get_model_name_arg("llama-3.2-3b")
        assert result == "meta-llama/Llama-3.2-3B-Instruct"

    def test_llama_1b_conversion(self):
        """Test Llama 3.2 1B model conversion."""
        result = get_model_name_arg("llama-3.2-1b")
        assert result == "meta-llama/Llama-3.2-1B-Instruct"

    def test_full_path_passthrough(self):
        """Test that full paths are passed through unchanged."""
        path = "meta-llama/Llama-3.2-3B-Instruct"
        result = get_model_name_arg(path)
        assert result == path

    def test_custom_model_passthrough(self):
        """Test that unknown models are passed through."""
        custom = "MyModel/Custom-7B"
        result = get_model_name_arg(custom)
        assert result == custom


class TestEfficiencyTracker:
    """Test efficiency tracking callback."""

    def test_initialization(self):
        """Test EfficiencyTracker initialization."""
        tracker = EfficiencyTracker()
        assert tracker.epoch_start_time is None
        assert tracker.epoch_times == []
        assert tracker.batch_times == []
        assert tracker.batch_start_time is None

    def test_on_epoch_begin(self):
        """Test epoch begin callback."""
        tracker = EfficiencyTracker()
        tracker.on_epoch_begin(None, None, None)
        assert tracker.epoch_start_time is not None

    def test_on_epoch_end(self):
        """Test epoch end callback."""
        import time

        tracker = EfficiencyTracker()
        tracker.on_epoch_begin(None, None, None)
        time.sleep(0.01)  # Sleep for 10ms
        tracker.on_epoch_end(None, None, None)
        assert len(tracker.epoch_times) == 1
        assert tracker.epoch_times[0] >= 0.01

    def test_on_step_begin_end(self):
        """Test batch step timing."""
        import time

        tracker = EfficiencyTracker()
        tracker.on_step_begin(None, None, None)
        time.sleep(0.01)  # Sleep for 10ms
        tracker.on_step_end(None, None, None)
        assert len(tracker.batch_times) == 1
        assert tracker.batch_times[0] >= 0.01

    def test_get_stats_empty(self):
        """Test get_stats with no data."""
        tracker = EfficiencyTracker()
        stats = tracker.get_stats()
        assert stats == {}

    def test_get_stats_with_data(self):
        """Test get_stats with epoch data."""

        tracker = EfficiencyTracker()

        # Add epoch times
        tracker.epoch_times = [1.0, 1.5, 0.9]
        tracker.batch_times = [0.1, 0.2, 0.15, 0.12]

        stats = tracker.get_stats()
        assert "avg_epoch_time_seconds" in stats
        assert "avg_batch_time_seconds" in stats
        assert "total_training_time_seconds" in stats

        # Check calculations
        assert abs(stats["avg_epoch_time_seconds"] - 1.1333) < 0.01
        assert abs(stats["total_training_time_seconds"] - 3.4) < 0.01


class TestCountTrainableParams:
    """Test trainable parameter counting."""

    def test_count_with_mock_model(self):
        """Test parameter counting with mock model."""
        import torch.nn as nn

        # Create a simple model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        total, trainable, pct = count_trainable_params(model)

        assert total > 0
        assert trainable > 0
        assert pct > 0
        assert pct <= 100

    def test_count_with_frozen_params(self):
        """Test parameter counting with frozen parameters."""
        import torch.nn as nn

        # Create a model with frozen parameters
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        total, trainable, pct = count_trainable_params(model)

        assert total > 0
        assert trainable > 0
        assert trainable < total
        assert pct < 100


class TestPeftMethodCreation:
    """Test PEFT method creation and configuration."""

    def test_peft_method_enum_from_value(self):
        """Test creating PeftMethod from value string."""
        method = PeftMethod("lora")
        assert method == PeftMethod.LORA

    def test_all_peft_methods_creatable(self):
        """Test that all PEFT methods can be created from value."""
        methods = [
            "lora",
            "prompt_tuning",
            "prefix_tuning",
            "lora_prompt",
            "lora_prefix",
        ]
        for method_name in methods:
            method = PeftMethod(method_name)
            assert method.value == method_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
