"""Inference for fact-checking models with numerical perturbations."""

import re
import json
from typing import Optional, List, Dict
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from collections import defaultdict
from datetime import datetime

from src.llm_finetune.config import prefix
from src.llm_finetune.data_loading_utils.hf_dataset_instructions import (
    load_fact_checking_data,
    FileLoadConfig,
    parse_file_config,
)

from unsloth import FastLanguageModel
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:  # vllm is optional (GPU-only)
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]

import os
import argparse
from tqdm import tqdm
import pandas as pd

# Valid labels for fact-checking (JSON format)
VALID_LABELS = ['{"label": true}', '{"label": false}']

# Timestamp for model naming
time_start = datetime.now()
model_name_timestamp = time_start.strftime("%Y%m%d_%H%M%S")


# TODO add more models
def get_model_name_arg(model_name):
    """Convert short model name to full HuggingFace path."""
    if model_name.startswith("Qwen3-8B") or model_name.startswith("Qwen3-8B"):
        return "unsloth/Qwen3-8B"
    elif model_name.startswith("Qwen3-32B") or model_name.startswith(
        "Qwen-3-32B"
    ):
        return "unsloth/Qwen3-32B"
    elif model_name.startswith("llama-3.2-3b"):
        return "unsloth/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b"):
        return "unsloth/Llama-3.2-1B-Instruct"
    return model_name


def extract_label_from_response(text: str) -> Optional[str]:
    """Extract the predicted label from model response.

    Expects JSON format: {"label": true} or {"label": false}
    Uses strict JSON parsing - only perfectly formatted responses are accepted.

    Returns:
        The extracted label in JSON format, or None if unparseable
    """

    original_text = text.strip()

    # Strip Qwen3 thinking tokens if present
    if "<think>" in original_text and "</think>" in original_text:
        # Remove everything between <think> and </think>
        text = re.sub(
            r"<think>.*?</think>", "", original_text, flags=re.DOTALL
        ).strip()
    elif "<think>" in original_text:
        # Handle case where thinking isn't closed - take content after </think>
        text = original_text.split("</think>")[-1].strip()
    else:
        text = original_text

    # Try to parse as JSON
    try:
        parsed = json.loads(text)

        # Validate against schema: must have "label" key with boolean value
        if isinstance(parsed, dict) and "label" in parsed:
            label_value = parsed["label"]

            # Ensure it's a boolean (not string "true"/"false")
            if isinstance(label_value, bool):
                # Return normalized JSON string format
                return f'{{"label": {str(label_value).lower()}}}'

        # If we get here, structure doesn't match schema
        return None

    except (json.JSONDecodeError, ValueError):
        # Could not parse as valid JSON
        return None


def inference(
    test_json_paths: Optional[List[str]] = None,
    test_file_configs: Optional[List[FileLoadConfig]] = None,
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    start_epoch: int = 0,
    use_vllm: bool = False,
    expand_perturbations: bool = True,
    max_seq_length: int = 16384,
    output_dir: str = f"{prefix}predictions/{model_name_timestamp}",
):
    """Run inference on test data."""

    print("\n" + "=" * 80)
    print("STARTING INFERENCE")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model from Epoch: {start_epoch}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Use vLLM: {use_vllm}")
    print(f"Expand perturbations: {expand_perturbations}")
    print("=" * 80 + "\n")

    # Configuration
    dtype = None
    load_in_4bit = True

    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"

    # 1. Check if the user provided a direct path to a local folder
    if os.path.isdir(model_name):
        print(f"📂 Mode: Direct local folder")
        model_name_arg = model_name

    # 2. If start_epoch > 0, look for the checkpoint in the training output directory
    elif start_epoch > 0:
        # Construct the path matching the training save logic:
        # {prefix}tuned_models/epoch-{epoch}/{name}/...

        # Path A: Correct path based on finetune() save_dir logic
        trained_path = f"{prefix}tuned_models/epoch-{start_epoch}/{model_name}{quant_status}/lora_numpert{rank_postfix}"

        if os.path.exists(trained_path):
            print(f"📂 Mode: Found training checkpoint (Standard)")
            model_name_arg = trained_path
        else:
            print(
                f"⚠️ Warning: Checkpoint for epoch {start_epoch} not found locally."
            )
            print(f"   Checked: {trained_path}")
            print(f"   Attempting to load '{model_name}' from Hugging Face...")
            model_name_arg = get_model_name_arg(model_name)

    # 3. Default to Hugging Face Hub or mapped alias
    else:
        print(f"☁️ Mode: Hugging Face / Alias")
        model_name_arg = get_model_name_arg(model_name)

    print(f"Loading model from: {model_name_arg}\n")

    # Load model
    if use_vllm:
        llm = LLM(
            model=model_name_arg,
            max_model_len=max_seq_length,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_arg, trust_remote_code=True
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_arg,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            fix_tokenizer=False,
        )
        FastLanguageModel.for_inference(model)

    # Load test data
    print("Loading test data...")
    test_data = load_fact_checking_data(
        json_paths=test_json_paths,
        file_configs=test_file_configs,
        split="test",
        expand_perturbations_flag=expand_perturbations,
        include_doc=True,
    )

    print(f"Test dataset size: {len(test_data)}")

    # Check sequence lengths
    print("\nChecking sequence lengths...")
    too_long = 0
    max_length_seen = 0

    for i, example in enumerate(test_data):
        formatted = tokenizer.apply_chat_template(
            example["conversations"], tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(formatted, return_tensors=None)
        token_count = len(tokens["input_ids"])
        max_length_seen = max(max_length_seen, token_count)

        if token_count > max_seq_length:
            too_long += 1
            if too_long <= 3:
                print(
                    f"⚠️  Example {i} too long: {token_count:,} tokens "
                    f"(claim_id: {example['claim_id']})"
                )

    print(f"\nMax sequence length in test set: {max_length_seen:,} tokens")

    if too_long > 0:
        print(
            f"⚠️  WARNING: {too_long}/{len(test_data)} examples exceed max_seq_length!"
        )
    else:
        print(f"✅ All examples fit within max_seq_length")

    # Prepare for predictions
    predictions = []  # Will contain None for parse failures
    true_labels = list(test_data["label"])
    raw_outputs = []  # Store raw outputs for debugging

    # Create output directory
    vllm_suffix = "-vllm" if use_vllm else ""
    pred_dir = f"{output_dir}/{model_name}{rank_postfix}{quant_status}-epoch-{start_epoch}{vllm_suffix}"
    os.makedirs(pred_dir, exist_ok=True)

    # Run inference
    print("\nRunning inference...")

    if use_vllm:
        sampling_params = SamplingParams(temperature=0, max_tokens=512, seed=42)
        prompts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in test_data["conversations"]
        ]

        outputs = llm.generate(prompts, sampling_params)

        for idx, output in enumerate(outputs):
            text = output.outputs[0].text
            raw_outputs.append(
                {
                    "index": idx,
                    "claim_id": test_data[idx]["claim_id"],
                    "perturbation_type": test_data[idx]["perturbation_type"],
                    "label": test_data[idx]["label"],
                    "full_prompt": prompts[idx],  # Full input
                    "raw_response": text,  # Model output only
                }
            )
            label = extract_label_from_response(text)
            predictions.append(label)

    else:
        for i in tqdm(range(len(test_data))):
            # Format the prompt
            prompt_text = tokenizer.apply_chat_template(
                test_data[i]["conversations"],
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = tokenizer.apply_chat_template(
                test_data[i]["conversations"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if isinstance(model_inputs, torch.Tensor):
                model_inputs = {"input_ids": model_inputs}

            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

            if "attention_mask" not in model_inputs:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    model_inputs["attention_mask"] = (
                        model_inputs["input_ids"] != 0
                    ).long()
                else:
                    model_inputs["attention_mask"] = (
                        model_inputs["input_ids"] != pad_id
                    ).long()

            # Store input length
            input_length = model_inputs["input_ids"].shape[1]

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.001,
            )

            # Decode only new tokens
            new_tokens = outputs[0][input_length:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

            raw_outputs.append(
                {
                    "index": i,
                    "claim_id": test_data[i]["claim_id"],
                    "perturbation_type": test_data[i]["perturbation_type"],
                    "label": test_data[i]["label"],
                    "full_prompt": prompt_text,  # Full input
                    "raw_response": decoded,  # Model output only
                }
            )

            label = extract_label_from_response(decoded)
            predictions.append(label)

    raw_outputs_path = f"{pred_dir}/inference_details.json"
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs, f, indent=2)
    print(
        f"Inference details (prompts + outputs) saved to: {raw_outputs_path}\n"
    )

    # Extract predictions for metrics
    predictions_only = [item["raw_response"] for item in raw_outputs]
    labels_extracted = [
        extract_label_from_response(resp) for resp in predictions_only
    ]

    # Separate valid predictions from parse failures
    valid_indices = [
        i for i, pred in enumerate(labels_extracted) if pred is not None
    ]
    parse_failure_indices = [
        i for i, pred in enumerate(labels_extracted) if pred is None
    ]

    valid_predictions = [labels_extracted[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]

    n_total = len(labels_extracted)
    n_valid = len(valid_predictions)
    n_failures = len(parse_failure_indices)

    # Report parse failures
    print(f"\n📊 Parse Results:")
    print(
        f"   Successfully parsed: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)"
    )
    print(
        f"   Parse failures: {n_failures}/{n_total} ({100*n_failures/n_total:.1f}%)"
    )

    if n_failures > 0:
        print(
            f"\n⚠️  WARNING: {n_failures} responses could not be parsed and are EXCLUDED from metrics."
        )
        print(
            "    Check inference_details.json and parse_failures.json for details."
        )

    # Calculate metrics (only on valid predictions)
    print("\n" + "=" * 80)
    print(f"RESULTS (on {n_valid}/{n_total} parseable responses)")
    print("=" * 80)

    if n_valid == 0:
        print("❌ ERROR: No valid predictions! Cannot compute metrics.")
        accuracy = 0.0
        f1_macro = 0.0
        f1_weighted = 0.0
    else:
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        f1_macro = f1_score(
            valid_true_labels,
            valid_predictions,
            average="macro",
            zero_division=0,
        )
        f1_weighted = f1_score(
            valid_true_labels,
            valid_predictions,
            average="weighted",
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                valid_true_labels, valid_predictions, labels=VALID_LABELS
            )
        )

    # Per-perturbation analysis (only on valid predictions)
    print("\nPer-Perturbation Type Analysis:")
    perturb_metrics: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "failures": 0}
    )

    # Count failures per perturbation type
    for i in parse_failure_indices:
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["failures"] += 1

    # Count correct/total for valid predictions
    for i in valid_indices:
        pred = labels_extracted[i]
        gt = true_labels[i]
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["total"] += 1
        if pred == gt:
            perturb_metrics[p_type]["correct"] += 1

    for p_type in sorted(perturb_metrics.keys()):
        metrics = perturb_metrics[p_type]
        total_with_failures = metrics["total"] + metrics["failures"]
        acc = (
            metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        )
        failure_note = (
            f" [+{metrics['failures']} unparsed]"
            if metrics["failures"] > 0
            else ""
        )
        print(
            f"  {p_type:30s}: {acc:.4f} ({metrics['correct']}/{metrics['total']}){failure_note}"
        )

    print("=" * 80 + "\n")

    # Save parse failures with full context
    if n_failures > 0:
        failures_info = [raw_outputs[i] for i in parse_failure_indices]

        failures_path = f"{pred_dir}/parse_failures.json"
        with open(failures_path, "w") as f:
            json.dump(failures_info, f, indent=2)
        print(
            f"Parse failure details (with full prompts) saved to: {failures_path}\n"
        )

    # Save predictions CSV
    results_df = pd.DataFrame(
        {
            "claim_id": test_data["claim_id"],
            "perturbation_type": test_data["perturbation_type"],
            "label": true_labels,
            "prediction": [
                p if p is not None else "PARSE_FAILURE"
                for p in labels_extracted
            ],
            "parsed_successfully": [p is not None for p in labels_extracted],
            "raw_response": [item["raw_response"] for item in raw_outputs],
        }
    )

    results_path = f"{pred_dir}/predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}\n")

    # Save metrics
    metrics_dict = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "total_examples": n_total,
        "valid_examples": n_valid,
        "parse_failures": n_failures,
        "parse_failure_rate": n_failures / n_total if n_total > 0 else 0,
        "valid_output_rate": n_valid / n_total if n_total > 0 else 0,
        "note": f"Metrics computed on {n_valid}/{n_total} parseable responses only.",
        "per_perturbation": {
            p_type: {
                "accuracy": m["correct"] / m["total"] if m["total"] > 0 else 0,
                "correct": m["correct"],
                "total": m["total"],
                "parse_failures": m["failures"],
            }
            for p_type, m in perturb_metrics.items()
        },
    }

    metrics_path = f"{pred_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_path}\n")

    # Save raw outputs for debugging parse failures
    if n_failures > 0:
        # Save detailed info about failures
        failures_info = [
            {
                "index": i,
                "claim_id": test_data[i]["claim_id"],
                "perturbation_type": test_data[i]["perturbation_type"],
                "label": true_labels[i],
                "raw_output": raw_outputs[i],
            }
            for i in parse_failure_indices
        ]
        failures_path = f"{pred_dir}/parse_failures.json"
        with open(failures_path, "w") as f:
            json.dump(failures_info, f, indent=2)
        print(f"Parse failure details saved to: {failures_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with fact-checking models"
    )

    # Data paths — supports per-file perturbation type control.
    # Format: path/to/file.json[:type1,type2,...] (see parse_file_config)
    parser.add_argument(
        "--test_json",
        nargs="+",
        required=True,
        help=("Path(s) to test JSON files. Same format as --train_json."),
    )

    # Model configuration
    parser.add_argument("--model", default=None, help="Model name")

    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epoch number of the trained model to load",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Maximum sequence length in tokens",
    )

    # Dataset configuration
    parser.add_argument(
        "--expand_perturbations",
        action="store_true",
        default=True,
        help="Expand perturbations in dataset",
    )

    # Inference options
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM for inference"
    )
    parser.add_argument(
        "--output_dir",
        default="predictions/numpert",
        help="Output directory for predictions",
    )

    args = parser.parse_args()

    # Parse file configs for test data
    test_configs = [parse_file_config(p) for p in args.test_json]

    inference(
        test_file_configs=test_configs,
        model_name=args.model,
        rank=args.rank,
        start_epoch=args.start_epoch,
        use_vllm=args.use_vllm,
        expand_perturbations=args.expand_perturbations,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )
