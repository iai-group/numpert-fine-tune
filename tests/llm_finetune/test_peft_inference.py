"""Unit tests for PEFT inference module."""

import pytest
from src.llm_finetune.peft_inference import InferenceEfficiencyTracker


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
