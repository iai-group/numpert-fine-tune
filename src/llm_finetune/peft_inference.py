"""PEFT-based inference for fact-checking models with efficiency tracking.

Reuses components from unsloth_inference.py while supporting multiple PEFT
methods. Tracks inference latency and throughput metrics.
"""

import argparse
import ast
import glob
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict

from tqdm import tqdm

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.llm_finetune.peft_finetune import setup_chat_template
from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.llm_finetune.utils.inference.inference_config_export import (
    build_inference_config,
    save_inference_config,
)
from src.llm_finetune.utils.inference.metrics import (
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
)
from src.prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_PROMPT,
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


_LABEL_RE = re.compile(r'\{\s*"label"\s*:\s*(true|false)\s*\}', re.IGNORECASE)


def extract_label_from_response(response_text: str) -> Optional[bool]:
    """Extract label from model response.

    Accepts {"label": true/false} anywhere in the response text, with or without
    surrounding content (Qwen3 thinking blocks, trailing explanation).

    Returns True/False, or None if no valid label found.
    """
    if "<think>" in response_text and "</think>" in response_text:
        response_text = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        )

    m = _LABEL_RE.search(response_text)
    if m is None:
        return None
    return m.group(1).lower() == "true"


def _build_prompt(example: dict, tokenizer) -> tuple[str, str, object]:
    """Build prompt string and extract claim text and ground-truth label."""
    label_true = example.get("label", None)
    convs_raw = example.get("conversations")
    if convs_raw is not None:
        convs = (
            ast.literal_eval(convs_raw)
            if isinstance(convs_raw, str)
            else convs_raw
        )
        messages = [m for m in convs if m.get("role") in ("system", "user")]
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        claim = user_content[:200]
    else:
        claim = example.get("claim", "")
        doc = example.get("doc", "")
        messages = [
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FACT_CHECK_USER_PROMPT.format(
                    examples="", claim=claim, evidence=doc
                ),
            },
        ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, claim, label_true


def _make_prediction_dict(
    example: dict,
    idx: int,
    claim: str,
    label_true,
    response: str,
    inference_latency: float,
    peak_memory_mb,
) -> tuple[dict, Optional[dict]]:
    """Build prediction and parse_failure dicts from a decoded response."""
    predicted_label = extract_label_from_response(response)

    if label_true is None:
        label_json = None
    elif isinstance(label_true, str) and label_true.strip().startswith("{"):
        label_json = label_true
    elif isinstance(label_true, str):
        label_json = json.dumps({"label": label_true.lower() == "true"})
    else:
        label_json = json.dumps({"label": bool(label_true)})

    prediction_json = (
        json.dumps({"label": predicted_label})
        if predicted_label is not None
        else None
    )

    prediction = {
        "claim_id": example.get("claim_id", example.get("id", f"sample_{idx}")),
        "claim": claim,
        "label": label_json,
        "prediction": prediction_json,
        "perturbation_type": example.get("perturbation_type", "unknown"),
        "source_file": example.get("source_file", ""),
        "raw_response": response,
        "latency_seconds": inference_latency,
        "peak_memory_mb": peak_memory_mb,
    }

    parse_failure = (
        {"claim_id": prediction["claim_id"], "response": response}
        if prediction_json is None
        else None
    )
    return prediction, parse_failure


def _process_inference_batch(
    examples: list[dict],
    start_idx: int,
    tokenizer,
    model,
    max_seq_length: int,
    track_memory: bool,
) -> list[tuple[dict, Optional[dict]]]:
    """Process a batch of samples with a single model.generate() call."""
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts, claims, labels_true = zip(
        *[_build_prompt(ex, tokenizer) for ex in examples]
    )

    # Left-padding is required for batched generation with decoder-only models.
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        list(prompts),
        truncation=True,
        max_length=max_seq_length,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
        )
        batch_latency = time.time() - start_time

    per_sample_latency = batch_latency / len(examples)
    peak_memory_mb = None
    if track_memory and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    results = []
    for i, example in enumerate(examples):
        generated_ids = outputs[i][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prediction, parse_failure = _make_prediction_dict(
            example,
            start_idx + i,
            claims[i],
            labels_true[i],
            response,
            per_sample_latency,
            peak_memory_mb,
        )
        results.append((prediction, parse_failure))

    return results


def _process_inference_sample(
    example: dict,
    idx: int,
    tokenizer,
    model,
    max_seq_length: int,
    track_memory: bool,
) -> tuple[dict, Optional[dict]]:
    """Process a single sample through inference."""
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompt, claim, label_true = _build_prompt(example, tokenizer)

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
        )
        inference_latency = time.time() - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    peak_memory_mb = None
    if track_memory and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return _make_prediction_dict(
        example,
        idx,
        claim,
        label_true,
        response,
        inference_latency,
        peak_memory_mb,
    )


def _save_inference_results(
    output_dir: str,
    predictions: list[dict],
    efficiency_stats: dict,
    parse_failures: list[dict],
    model_path: str,
    max_seq_length: int,
) -> None:
    """Save inference results using the shared metrics pipeline."""
    # Compute full structured metrics report
    metrics_report = compute_metrics(predictions)
    print_metrics_report(metrics_report)

    print("\nEfficiency Metrics:")
    if efficiency_stats:
        print(f"  Avg latency: {efficiency_stats['avg_latency_ms']:.2f} ms")
        print(
            f"  Throughput: "
            f"{efficiency_stats['avg_throughput_samples_per_sec']:.2f} samples/sec"  # noqa: E501
        )
        if "avg_peak_memory_mb" in efficiency_stats:
            print(
                f"  Avg peak memory: {efficiency_stats['avg_peak_memory_mb']:.1f} MB"  # noqa: E501
            )

    print("\nSaving results...")

    # Save predictions JSON
    predictions_file = Path(output_dir) / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Predictions: {predictions_file}")

    # Save metrics (structured format matching other inference pipelines)
    metrics_path = save_metrics_report(metrics_report, output_dir)
    print(f"  Metrics: {metrics_path}")

    # Save parse failures
    if parse_failures:
        failures_file = Path(output_dir) / "parse_failures.json"
        with open(failures_file, "w") as f:
            json.dump(parse_failures, f, indent=2)
        print(f"  Parse failures: {failures_file}")

    # Save predictions CSV
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "claim_id": p["claim_id"],
                "perturbation_type": p.get("perturbation_type", ""),
                "source_file": p.get("source_file", ""),
                "label": p["label"],
                "prediction": p["prediction"]
                if p["prediction"] is not None
                else "PARSE_FAILURE",
                "parsed_successfully": p["prediction"] is not None,
                "raw_response": p.get("raw_response", ""),
            }
            for p in predictions
        ]
    )
    predictions_csv = Path(output_dir) / "predictions.csv"
    df.to_csv(predictions_csv, index=False)
    print(f"  Predictions CSV: {predictions_csv}")

    # Save inference config
    inference_config = build_inference_config(
        model_name="peft_model",
        model_path=model_path,
        max_seq_length=max_seq_length,
    )
    config_file = save_inference_config(inference_config, output_dir)
    print(f"  Inference config: {config_file}")


def run_inference(
    model_path: str,
    test_json_path: str,
    output_dir: str = "./inference_results",
    batch_size: int = 8,
    max_seq_length: int = 16384,
    track_memory: bool = False,
    sample_frac: float = 1.0,
):
    """Run inference on test set with efficiency tracking.

    Args:
        model_path: Path to fine-tuned PEFT model
        test_json_path: Path to test data JSON file
        output_dir: Directory to save results
        batch_size: Batch size for inference
        max_seq_length: Max sequence length
        track_memory: Whether to track GPU memory usage
        sample_frac: Fraction of test data to use (0 < frac <= 1); useful for smoke tests # noqa: E501
    """
    print("\n" + "=" * 80)
    print("STARTING PEFT INFERENCE")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_json_path}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Setup
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
    setup_chat_template(tokenizer, model_path)

    print("Model loaded successfully")

    # Load test data using the shared pipeline (adds perturbation_type, source_file) # noqa: E501
    print(f"Loading test data from {test_json_path}...")
    matched_files = sorted(glob.glob(test_json_path))
    if not matched_files:
        raise FileNotFoundError(f"No files matched: {test_json_path}")
    test_dataset = load_split(json_paths=matched_files, split="test")
    test_data = [test_dataset[i] for i in range(len(test_dataset))]
    if sample_frac < 1.0:
        import math

        n = max(1, math.ceil(len(test_data) * sample_frac))
        test_data = test_data[:n]
        print(f"Sampling {n}/{len(test_dataset)} examples ({sample_frac:.0%})")
    print(
        f"Loaded {len(test_data)} test examples from {len(matched_files)} file(s)"  # noqa: E501
    )

    # Prepare output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run inference
    predictions = []
    parse_failures = []

    print("\nRunning inference...\n")

    parse_fail_count = 0
    with tqdm(total=len(test_data), unit="claim", dynamic_ncols=True) as pbar:
        for batch_start in range(0, len(test_data), batch_size):
            batch = test_data[batch_start : batch_start + batch_size]
            batch_results = _process_inference_batch(
                batch,
                batch_start,
                tokenizer,
                model,
                max_seq_length,
                track_memory,
            )

            for prediction, parse_failure in batch_results:
                predictions.append(prediction)
                efficiency_tracker.add_sample(
                    prediction["latency_seconds"],
                    prediction.get("peak_memory_mb"),
                )
                if parse_failure:
                    parse_failures.append(parse_failure)
                    parse_fail_count += 1

            pbar.update(len(batch))
            last_latency = batch_results[-1][0]["latency_seconds"]
            pbar.set_postfix(
                latency=f"{last_latency*1000:.0f}ms",  # noqa: E226
                parse_fails=parse_fail_count,
            )

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)

    # Get efficiency stats
    efficiency_stats = efficiency_tracker.get_stats()

    # Compute metrics, print, and save
    _save_inference_results(
        output_dir,
        predictions,
        efficiency_stats,
        parse_failures,
        model_path,
        max_seq_length,
    )

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
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of test data to use (e.g. 0.05 for 5%% smoke test)",
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        test_json_path=args.test_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        track_memory=args.track_memory,
        sample_frac=args.sample_frac,
    )
