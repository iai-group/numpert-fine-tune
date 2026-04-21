"""PEFT-based inference for fact-checking models with efficiency tracking.

Reuses components from unsloth_inference.py while supporting multiple PEFT
methods. Tracks inference latency and throughput metrics.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import csv

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.llm_finetune.utils.inference.inference_config_export import (
    build_inference_config,
    save_inference_config,
)


class InferenceEfficiencyTracker:
    """Track efficiency metrics during inference."""

    def __init__(self) -> None:
        """Initialize inference efficiency tracker."""
        self.latencies: list[float] = []  # Per-sample latency in seconds
        self.throughputs: list[float] = []  # Samples/second
        self.memory_usage: list[float] = []  # Peak GPU memory in MB

    def add_sample(
        self,
        latency: float,
        peak_memory_mb: Optional[float] = None,
    ):
        """Record a sample's inference latency and memory."""
        self.latencies.append(latency)
        if peak_memory_mb is not None:
            self.memory_usage.append(peak_memory_mb)

    def get_stats(self) -> Dict[str, float]:
        """Get aggregated efficiency statistics."""
        if not self.latencies:
            return {}

        avg_latency = sum(self.latencies) / len(self.latencies)
        min_latency = min(self.latencies)
        max_latency = max(self.latencies)
        avg_throughput = 1.0 / avg_latency  # samples/second

        stats = {
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "avg_throughput_samples_per_sec": avg_throughput,
            "total_samples": len(self.latencies),
        }

        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            stats["avg_peak_memory_mb"] = avg_memory
            stats["max_peak_memory_mb"] = max_memory

        return stats


def extract_label_from_response(response_text: str) -> Optional[bool]:
    """Extract label from model response.

    Expects JSON format: {"label": true} or {"label": false}
    Returns None if unparseable.
    """
    # Strip Qwen3 thinking tokens if present
    if "<think>" in response_text:
        response_text = response_text.split("</think>")[-1]

    response_text = response_text.strip()

    # Try to extract JSON
    try:
        # Find JSON object
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = response_text[start_idx:end_idx]
        parsed = json.loads(json_str)

        if "label" in parsed:
            label_val = parsed["label"]
            if isinstance(label_val, bool):
                return label_val
            elif isinstance(label_val, str):
                return label_val.lower() == "true"
            else:
                return bool(label_val)

    except (json.JSONDecodeError, ValueError):
        pass

    return None


def run_inference(
    model_path: str,
    test_json_path: str,
    output_dir: str = "./inference_results",
    batch_size: int = 8,
    max_seq_length: int = 16384,
    track_memory: bool = False,
):
    """Run inference on test set with efficiency tracking.

    Args:
        model_path: Path to fine-tuned PEFT model
        test_json_path: Path to test data JSON file
        output_dir: Directory to save results
        batch_size: Batch size for inference
        max_seq_length: Max sequence length
        track_memory: Whether to track GPU memory usage
    """
    print("\n" + "=" * 80)
    print("STARTING PEFT INFERENCE")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_json_path}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    efficiency_tracker = InferenceEfficiencyTracker()

    # Load model
    print("Loading PEFT model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
        if torch.cuda.is_available()
        else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Model loaded successfully")

    # Load test data
    print(f"Loading test data from {test_json_path}...")
    with open(test_json_path) as f:
        content = f.read().strip()
        # Handle both JSON array and JSONL formats
        if content.startswith("["):
            # JSON array format: [{ ... }, { ... }]
            test_data = json.loads(content)
        else:
            # JSONL format: one object per line
            test_data = [
                json.loads(line) for line in content.split("\n") if line.strip()
            ]
    print(f"Loaded {len(test_data)} test examples")

    # Prepare output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run inference
    predictions = []
    parse_failures = []
    total_inference_time: float = 0.0

    print("\nRunning inference...\n")

    for idx, example in enumerate(test_data):
        if (idx + 1) % max(1, len(test_data) // 10) == 0:
            print(f"  Processing {idx + 1}/{len(test_data)} samples...")

        # Get peak memory before
        if track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Prepare input
        claim = example.get("claim", "")
        doc = example.get("doc", "")
        label_true = example.get("label", None)

        prompt = (
            "Given the claim and evidence below, determine if the claim is "
            f"true or false.\n\nClaim: {claim}\n\nEvidence: {doc}"
        )

        # Tokenize
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Inference timing
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )
            inference_latency = time.time() - start_time

        # Decode - only get the generated tokens (skip input)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract label
        predicted_label = extract_label_from_response(response)

        # Track memory
        peak_memory_mb = None
        if track_memory and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        efficiency_tracker.add_sample(inference_latency, peak_memory_mb)

        # Store prediction
        parse_success = predicted_label is not None
        predictions.append(
            {
                "claim_id": example.get("id", f"sample_{idx}"),
                "claim": claim,
                "ground_truth": label_true,
                "predicted_label": predicted_label,
                "parse_success": parse_success,
                "response": response,
                "latency_seconds": inference_latency,
            }
        )

        if not parse_success:
            parse_failures.append(
                {
                    "claim_id": example.get("id", f"sample_{idx}"),
                    "response": response,
                }
            )

        total_inference_time += inference_latency

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)

    # Compute metrics
    print("\nComputing metrics...")
    metrics_dict: Dict[str, Any] = {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "parse_success_rate": sum(
            1 for p in predictions if p.get("parse_success", False)
        )
        / len(predictions)
        if predictions
        else 0.0,
    }

    # Get efficiency stats
    efficiency_stats = efficiency_tracker.get_stats()

    # Print results
    print("\nAccuracy Metrics:")
    print(f"  Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"  F1 Score (macro): {metrics_dict['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics_dict['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics_dict['recall_macro']:.4f}")
    print(f"  Parse success rate: {metrics_dict['parse_success_rate']:.4f}")

    print("\nEfficiency Metrics:")
    print(f"  Avg latency: {efficiency_stats['avg_latency_ms']:.2f} ms")
    print(
        f"  Throughput: "
        f"{efficiency_stats['avg_throughput_samples_per_sec']:.2f} samples/sec"
    )
    if "avg_peak_memory_mb" in efficiency_stats:
        print(
            f"  Avg peak memory: {efficiency_stats['avg_peak_memory_mb']:.1f} MB"  # noqa: E501
        )

    # Save results
    print("\nSaving results...")

    # Save predictions
    predictions_file = Path(output_dir) / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Predictions: {predictions_file}")

    # Save metrics
    metrics_file = Path(output_dir) / "metrics.json"
    metrics_output = {
        "accuracy_metrics": metrics_dict,
        "efficiency_metrics": efficiency_stats,
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  Metrics: {metrics_file}")

    # Save parse failures
    if parse_failures:
        failures_file = Path(output_dir) / "parse_failures.json"
        with open(failures_file, "w") as f:
            json.dump(parse_failures, f, indent=2)
        print(f"  Parse failures: {failures_file}")

    # Save predictions CSV for analysis
    predictions_csv = Path(output_dir) / "predictions.csv"
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "ground_truth",
                "predicted_label",
                "parse_success",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        for p in predictions:
            writer.writerow(
                {
                    "claim_id": p["claim_id"],
                    "ground_truth": p["ground_truth"],
                    "predicted_label": p["predicted_label"],
                    "parse_success": p["parse_success"],
                    "latency_seconds": p["latency_seconds"],
                }
            )
    print(f"  Predictions CSV: {predictions_csv}")

    # Save inference config
    inference_config = build_inference_config(
        model_name="peft_model",
        model_path=model_path,
        max_seq_length=max_seq_length,
    )
    config_file = save_inference_config(inference_config, output_dir)
    print(f"  Inference config: {config_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on PEFT fine-tuned models"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned PEFT model",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Max sequence length",
    )
    parser.add_argument(
        "--track_memory",
        type=bool,
        default=False,
        help="Track GPU memory usage",
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        test_json_path=args.test_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        track_memory=args.track_memory,
    )
