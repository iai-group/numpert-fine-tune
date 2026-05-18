"""Comparison script for running PEFT methods side-by-side.

This script trains multiple PEFT methods on the same data and generates
comparison reports on training efficiency, inference efficiency, and accuracy.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

import pandas as pd


class PeftMethod(str, Enum):
    """Supported PEFT methods."""

    LORA = "lora"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    LORA_PROMPT = "lora_prompt"
    LORA_PREFIX = "lora_prefix"


def run_training(
    method: PeftMethod,
    train_json: str,
    val_json: str,
    model: str,
    rank: int,
    num_virtual_tokens: int,
    num_train_epochs: int,
    batch_size: int,
    grad_accum: int,
    output_dir: str,
    sample_percent: float = 100,
) -> Dict[str, Any]:
    """Run training for a specific PEFT method.

    Returns:
        Dictionary with training config and paths
    """
    print(f"\n{'=' * 80}")
    print(f"Training {method.value.upper()}")
    print(f"{'=' * 80}\n")

    cmd = [
        "python",
        "-m",
        "src.llm_finetune.peft_finetune",
        "--train_json",
        train_json,
        "--val_json",
        val_json,
        "--model",
        model,
        "--peft_method",
        method.value,
        "--rank",
        str(rank),
        "--num_virtual_tokens",
        str(num_virtual_tokens),
        "--num_train_epochs",
        str(num_train_epochs),
        "--batch_size",
        str(batch_size),
        "--grad_accum",
        str(grad_accum),
        "--sample_percent",
        str(sample_percent),
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        return {"status": "success", "method": method.value}
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {method.value}")
        print(f"Error: {e}")
        return {"status": "failed", "method": method.value, "error": str(e)}


def run_inference(
    model_path: str,
    test_json: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run inference on a trained model.

    Returns:
        Dictionary with inference results
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference: {model_path}")
    print(f"{'=' * 80}\n")

    cmd = [
        "python",
        "-m",
        "src.llm_finetune.peft_inference",
        "--model_path",
        model_path,
        "--test_json",
        test_json,
        "--output_dir",
        output_dir,
        "--track_memory",
        "True",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)

        # Load results
        metrics_file = Path(output_dir) / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            return {"status": "success", "metrics": metrics}
        else:
            return {"status": "failed", "error": "Metrics file not found"}

    except subprocess.CalledProcessError as e:
        print("❌ Inference failed")
        print(f"Error: {e}")
        return {"status": "failed", "error": str(e)}


def create_comparison_report(
    results: Dict[str, Any],
    output_file: str,
) -> None:
    """Create a comparison report from results.

    Args:
        results: Dictionary with results for each method
        output_file: Path to save report
    """
    print(f"\n{'=' * 80}")
    print("Creating Comparison Report")
    print(f"{'=' * 80}\n")

    # Extract data for each method
    report_data = []

    for method_name, method_results in results.items():
        if method_results.get("status") != "success":
            continue

        training_config = method_results.get("training_config", {})
        inference_metrics = method_results.get("inference_metrics", {})

        # Accuracy metrics
        accuracy_metrics = inference_metrics.get("accuracy_metrics", {})
        # Efficiency metrics
        efficiency_metrics = inference_metrics.get("efficiency_metrics", {})
        # Training efficiency
        training_efficiency = training_config.get("training_efficiency", {})

        row = {
            "Method": method_name,
            # Training
            "Total Params": training_config.get("total_parameters", 0),
            "Trainable Params": training_config.get("trainable_parameters", 0),
            "Trainable %": (
                f"{training_config.get('trainable_percentage', 0):.2f}%"
            ),
            # Accuracy
            "Accuracy": f"{accuracy_metrics.get('accuracy', 0):.4f}",
            "F1 (macro)": f"{accuracy_metrics.get('f1_macro', 0):.4f}",
            "Precision": f"{accuracy_metrics.get('precision_macro', 0):.4f}",
            "Recall": f"{accuracy_metrics.get('recall_macro', 0):.4f}",
            # Inference efficiency
            "Avg Latency (ms)": (
                f"{efficiency_metrics.get('avg_latency_ms', 0):.2f}"
            ),
            "Throughput (samples/s)": (
                f"{efficiency_metrics.get('avg_throughput_samples_per_sec', 0):.2f}"  # noqa: E501
            ),
            "Peak Memory (MB)": (
                f"{efficiency_metrics.get('max_peak_memory_mb', 0):.1f}"
            ),
            # Training efficiency
            "Avg Epoch Time (s)": (
                f"{training_efficiency.get('avg_epoch_time_seconds', 0):.2f}"
            ),
            "Total Training Time (s)": (
                f"{training_efficiency.get('total_training_time_seconds', 0):.2f}"  # noqa: E501
            ),
        }

        report_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(report_data)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Report saved to: {output_file}\n")

    # Print table
    print(df.to_string(index=False))
    print()


def main(
    train_json: str,
    val_json: str,
    test_json: str,
    methods: List[str],
    model: str = "Qwen/Qwen2.5-8B-Instruct",
    rank: int = 32,
    num_virtual_tokens: int = 20,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 1,
    output_base_dir: str = "./comparison_results",
    sample_percent: float = 100,
):
    """Run comparison of PEFT methods.

    Args:
        train_json: Path to training data
        val_json: Path to validation data
        test_json: Path to test data
        methods: List of PEFT methods to compare
        model: Model name/path
        rank: LoRA rank
        num_virtual_tokens: Virtual tokens for prompt/prefix tuning
        num_train_epochs: Number of training epochs
        batch_size: Batch size
        grad_accum: Gradient accumulation steps
        output_base_dir: Base directory for outputs
        sample_percent: Percentage of training data to use
    """
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_base_dir) / f"comparison_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("PEFT Methods Comparison")
    print(f"{'=' * 80}")
    print(f"Timestamp: {timestamp}")
    print(f"Model: {model}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Output directory: {base_dir}")
    print(f"{'=' * 80}\n")

    results = {}

    # Train each method
    for method_str in methods:
        method = PeftMethod(method_str)
        method_dir = base_dir / method.value

        # Run training
        training_result = run_training(
            method=method,
            train_json=train_json,
            val_json=val_json,
            model=model,
            rank=rank,
            num_virtual_tokens=num_virtual_tokens,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            output_dir=str(method_dir),
            sample_percent=sample_percent,
        )

        if training_result["status"] != "success":
            results[method.value] = training_result
            continue

        # Load training config
        # (In a real implementation, this would parse the saved config)
        results[method.value] = {
            "status": "success",
            "training_config": {},  # Populated from training
        }

        # Run inference
        model_path = method_dir / "checkpoints"
        inference_result = run_inference(
            model_path=str(model_path),
            test_json=test_json,
            output_dir=str(method_dir / "inference"),
        )

        results[method.value]["inference_metrics"] = inference_result.get(
            "metrics", {}
        )

    # Create comparison report
    report_file = base_dir / "comparison_report.csv"
    create_comparison_report(results, str(report_file))

    print(f"\n{'=' * 80}")
    print("Comparison Complete!")
    print(f"Results saved to: {base_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare PEFT fine-tuning methods"
    )

    parser.add_argument(
        "--train_json",
        type=str,
        required=True,
        help="Training data JSON file",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        required=True,
        help="Validation data JSON file",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Test data JSON file",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="lora,prompt_tuning,prefix_tuning,lora_prompt,lora_prefix",
        help="Comma-separated list of PEFT methods to compare",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="Number of virtual tokens",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Output directory",
    )
    parser.add_argument(
        "--sample_percent",
        type=float,
        default=100,
        help="Percentage of training data to use (e.g., 10 for 10% sample)",
    )

    args = parser.parse_args()
    methods = args.methods.split(",")

    main(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        methods=methods,
        model=args.model,
        rank=args.rank,
        num_virtual_tokens=args.num_virtual_tokens,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        output_base_dir=args.output_dir,
        sample_percent=args.sample_percent,
    )
