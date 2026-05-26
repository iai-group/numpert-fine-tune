"""Unit tests for PEFT inference module."""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from src.llm_finetune.peft_inference import (
    InferenceEfficiencyTracker,
    extract_label_from_response,
)


class TestInferenceEfficiencyTracker:
    """Test inference efficiency tracking."""

    def test_initialization(self):
        """Test InferenceEfficiencyTracker initialization."""
        tracker = InferenceEfficiencyTracker()
        assert tracker.latencies == []
        assert tracker.throughputs == []
        assert tracker.memory_usage == []

    def test_add_sample_latency_only(self):
        """Test adding sample with latency only."""
        tracker = InferenceEfficiencyTracker()
        tracker.add_sample(0.05)
        assert len(tracker.latencies) == 1
        assert tracker.latencies[0] == 0.05

    def test_add_sample_with_memory(self):
        """Test adding sample with latency and memory."""
        tracker = InferenceEfficiencyTracker()
        tracker.add_sample(0.05, peak_memory_mb=512.5)
        assert len(tracker.latencies) == 1
        assert tracker.latencies[0] == 0.05
        assert len(tracker.memory_usage) == 1
        assert tracker.memory_usage[0] == 512.5

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        tracker = InferenceEfficiencyTracker()
        samples = [
            (0.04, 500.0),
            (0.05, 510.0),
            (0.03, 490.0),
            (0.06, 520.0),
        ]

        for latency, memory in samples:
            tracker.add_sample(latency, peak_memory_mb=memory)

        assert len(tracker.latencies) == 4
        assert len(tracker.memory_usage) == 4

    def test_get_stats_empty(self):
        """Test get_stats with no data."""
        tracker = InferenceEfficiencyTracker()
        stats = tracker.get_stats()
        assert stats == {}

    def test_get_stats_single_sample(self):
        """Test get_stats with single sample."""
        tracker = InferenceEfficiencyTracker()
        tracker.add_sample(0.05, peak_memory_mb=512.0)
        stats = tracker.get_stats()

        assert "avg_latency_ms" in stats
        assert "min_latency_ms" in stats
        assert "max_latency_ms" in stats
        assert "avg_throughput_samples_per_sec" in stats
        assert "avg_peak_memory_mb" in stats
        assert "max_peak_memory_mb" in stats

    def test_get_stats_calculations(self):
        """Test that stats are calculated correctly."""
        tracker = InferenceEfficiencyTracker()

        # Add samples: 0.05, 0.04, 0.06 seconds (50, 40, 60 ms)
        tracker.add_sample(0.05, peak_memory_mb=500.0)
        tracker.add_sample(0.04, peak_memory_mb=510.0)
        tracker.add_sample(0.06, peak_memory_mb=490.0)

        stats = tracker.get_stats()

        # Latency in milliseconds
        assert abs(stats["min_latency_ms"] - 40.0) < 0.1
        assert abs(stats["max_latency_ms"] - 60.0) < 0.1
        assert abs(stats["avg_latency_ms"] - 50.0) < 0.1

        # Throughput: 1 / avg_latency (in seconds)
        expected_throughput = 1.0 / 0.05  # 20 samples/sec
        assert (
            abs(stats["avg_throughput_samples_per_sec"] - expected_throughput)
            < 0.1
        )

        # Memory statistics
        assert abs(stats["avg_peak_memory_mb"] - 500.0) < 0.1
        assert abs(stats["max_peak_memory_mb"] - 510.0) < 0.1

    def test_latency_in_milliseconds(self):
        """Test that latency is properly converted to milliseconds."""
        tracker = InferenceEfficiencyTracker()
        tracker.add_sample(0.001)  # 1ms = 0.001 seconds

        stats = tracker.get_stats()
        assert abs(stats["avg_latency_ms"] - 1.0) < 0.01

    def test_throughput_calculation(self):
        """Test throughput is calculated as 1/latency."""
        tracker = InferenceEfficiencyTracker()
        # If latency is 0.1 seconds, throughput should be 10 samples/sec
        tracker.add_sample(0.1)

        stats = tracker.get_stats()
        assert abs(stats["avg_throughput_samples_per_sec"] - 10.0) < 0.1

    def test_high_throughput(self):
        """Test with very fast inferences."""
        tracker = InferenceEfficiencyTracker()
        # 1ms latency = 1000 samples/sec throughput
        for _ in range(10):
            tracker.add_sample(0.001)

        stats = tracker.get_stats()
        assert abs(stats["avg_throughput_samples_per_sec"] - 1000.0) < 10.0

    def test_variable_latency(self):
        """Test with variable latencies."""
        tracker = InferenceEfficiencyTracker()
        latencies = [0.02, 0.05, 0.03, 0.04, 0.08]

        for lat in latencies:
            tracker.add_sample(lat)

        stats = tracker.get_stats()
        avg_latency = sum(latencies) / len(latencies)
        assert abs(stats["avg_latency_ms"] - avg_latency * 1000) < 0.1
        assert abs(stats["min_latency_ms"] - min(latencies) * 1000) < 0.1
        assert abs(stats["max_latency_ms"] - max(latencies) * 1000) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestExtractLabelFromResponse:
    """Test label extraction from model responses."""

    def test_extract_valid_json_true(self):
        """Test extracting label: true."""
        response = '{"label": true}'
        assert extract_label_from_response(response) is True

    def test_extract_valid_json_false(self):
        """Test extracting label: false."""
        response = '{"label": false}'
        assert extract_label_from_response(response) is False

    def test_extract_with_extra_text_before(self):
        """Test extracting JSON with extra text before."""
        response = 'The answer is: {"label": true}'
        assert extract_label_from_response(response) is True

    def test_extract_with_extra_text_after(self):
        """Test extracting JSON with extra text after."""
        response = '{"label": false} That is the answer.'
        assert extract_label_from_response(response) is False

    def test_extract_with_thinking_tokens(self):
        """Test extracting with Qwen3 thinking tokens."""
        response = '<think>Let me think about this...</think>\n{"label": true}'
        assert extract_label_from_response(response) is True

    def test_extract_string_true(self):
        """Test extracting string 'true' value."""
        response = '{"label": "true"}'
        assert extract_label_from_response(response) is True

    def test_extract_string_false(self):
        """Test extracting string 'false' value."""
        response = '{"label": "false"}'
        assert extract_label_from_response(response) is False

    def test_extract_string_uppercase_true(self):
        """Test extracting uppercase string 'True'."""
        response = '{"label": "True"}'
        assert extract_label_from_response(response) is True

    def test_extract_invalid_json(self):
        """Test invalid JSON returns None."""
        response = "not valid json"
        assert extract_label_from_response(response) is None

    def test_extract_no_label_field(self):
        """Test JSON without 'label' field returns None."""
        response = '{"result": true}'
        assert extract_label_from_response(response) is None

    def test_extract_empty_string(self):
        """Test empty string returns None."""
        response = ""
        assert extract_label_from_response(response) is None

    def test_extract_numeric_label(self):
        """Test numeric label values."""
        assert extract_label_from_response('{"label": 1}') is True
        assert extract_label_from_response('{"label": 0}') is False


def _make_fake_dataset(claim_id: str = "test_1", label: bool = True) -> list:
    """Return a one-item list matching what load_split returns."""
    convs = [
        {"role": "system", "content": "You are a fact-checker."},
        {"role": "user", "content": "Claim: Test claim\nEvidence: Test doc"},
        {"role": "assistant", "content": json.dumps({"label": label})},
    ]
    return [
        {
            "conversations": str(convs),
            "claim_id": claim_id,
            "label": json.dumps({"label": label}),
            "perturbation_type": "original_normalized",
            "source_file": "test.json",
        }
    ]


class TestRunInferenceIntegration:
    """Integration tests for run_inference."""

    @patch("src.llm_finetune.peft_inference.load_split")
    @patch("src.llm_finetune.peft_inference.setup_chat_template")
    @patch("src.llm_finetune.peft_inference.AutoPeftModelForCausalLM")
    @patch("src.llm_finetune.peft_inference.AutoTokenizer")
    @patch("src.llm_finetune.peft_inference.build_inference_config")
    @patch("src.llm_finetune.peft_inference.save_inference_config")
    def test_run_inference_with_sample_data(
        self,
        mock_save_config,
        mock_build_config,
        mock_tokenizer_class,
        mock_model_class,
        mock_setup_chat,
        mock_load_split,
    ):
        """Test run_inference with mocked model and tokenizer."""
        from src.llm_finetune.peft_inference import run_inference

        mock_load_split.return_value = _make_fake_dataset("test_1", label=True)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=[1, 2, 3])
        mock_model.generate.return_value = [mock_output]
        mock_tokenizer.decode.return_value = '{"label": true}'
        mock_build_config.return_value = {}
        mock_save_config.return_value = "config.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text("[]")
            output_dir = Path(tmpdir) / "output"

            run_inference(
                model_path="test_model",
                test_json_path=str(test_file),
                output_dir=str(output_dir),
                track_memory=False,
            )

            assert (output_dir / "predictions.json").exists()
            assert (output_dir / "metrics.json").exists()
            assert (output_dir / "predictions.csv").exists()
            mock_model_class.from_pretrained.assert_called_once()

            with open(output_dir / "predictions.json") as f:
                predictions = json.load(f)
            assert len(predictions) == 1
            assert predictions[0]["claim_id"] == "test_1"
            assert predictions[0]["label"] is not None

    @patch("src.llm_finetune.peft_inference.load_split")
    @patch("src.llm_finetune.peft_inference.setup_chat_template")
    @patch("src.llm_finetune.peft_inference.AutoPeftModelForCausalLM")
    @patch("src.llm_finetune.peft_inference.AutoTokenizer")
    @patch("src.llm_finetune.peft_inference.build_inference_config")
    @patch("src.llm_finetune.peft_inference.save_inference_config")
    def test_run_inference_handles_parse_failures(
        self,
        mock_save_config,
        mock_build_config,
        mock_tokenizer_class,
        mock_model_class,
        mock_setup_chat,
        mock_load_split,
    ):
        """Test that parse failures are handled correctly."""
        from src.llm_finetune.peft_inference import run_inference

        mock_load_split.return_value = _make_fake_dataset("test_1", label=True)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=[1, 2, 3])
        mock_model.generate.return_value = [mock_output]
        mock_tokenizer.decode.return_value = "invalid response"
        mock_build_config.return_value = {}
        mock_save_config.return_value = "config.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text("[]")
            output_dir = Path(tmpdir) / "output"

            run_inference(
                model_path="test_model",
                test_json_path=str(test_file),
                output_dir=str(output_dir),
                track_memory=False,
            )

            assert (output_dir / "parse_failures.json").exists()
            with open(output_dir / "parse_failures.json") as f:
                failures = json.load(f)
            assert len(failures) == 1
            assert failures[0]["claim_id"] == "test_1"
