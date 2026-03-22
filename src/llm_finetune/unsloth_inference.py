"""Inference for fact-checking models with numerical perturbations."""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional, List

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.llm_finetune.config import prefix
from src.llm_finetune.utils.data_loading.hf_dataset_instructions import (
    load_fact_checking_data,
    FileLoadConfig,
    parse_file_config,
)
from src.llm_finetune.utils.inference.inference_config_export import (
    build_inference_config,
    save_inference_config,
)
from src.llm_finetune.utils.inference.metrics import (
    build_result_records,
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
    save_parse_failures,
    save_predictions_csv,
)
from src.llm_finetune.utils.naming import (
    build_output_dir_name,
    generate_run_name,
)


def get_model_name_arg(model_name):
    """Convert short model name to full HuggingFace path."""
    if model_name.startswith("Qwen3-8B") or model_name.startswith(
        "unsloth/Qwen3-8B"
    ):
        return "unsloth/Qwen3-8B"
    elif model_name.startswith("Qwen3-32B") or model_name.startswith(
        "unsloth/Qwen3-32B"
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


def inference(  # noqa: C901
    test_json_paths: Optional[List[str]] = None,
    test_file_configs: Optional[List[FileLoadConfig]] = None,
    model_name: str = "unsloth/Qwen3-8B",
    rank: int = 32,
    start_epoch: int = 0,
    expand_perturbations: bool = True,
    max_seq_length: int = 16384,
    output_dir: Optional[str] = None,
    inference_unperturbed_claims: bool = False,
):
    """Run inference on test data."""
    time_start = datetime.now()

    if output_dir is None:
        predictions_dir = f"{prefix}predictions"
        run_name = generate_run_name(
            model_name,
            timestamp=time_start,
            base_path=predictions_dir,
        )
        output_dir = f"{predictions_dir}/{run_name}"

    if inference_unperturbed_claims:
        expand_perturbations = False

    print("\n" + "=" * 80)
    print(
        "STARTING INFERENCE"
        + (" (UNPERTURBED CLAIMS)" if inference_unperturbed_claims else "")
    )
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model from Epoch: {start_epoch}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Expand perturbations: {expand_perturbations}")
    if inference_unperturbed_claims:
        print(
            "Unperturbed mode: using only normalized_claim (no perturbations)"
        )
    print("=" * 80 + "\n")

    # Configuration
    dtype = None
    load_in_4bit = True

    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"

    # 1. Check if the user provided a direct path to a local folder
    if os.path.isdir(model_name):
        print("📂 Mode: Direct local folder")
        model_name_arg = model_name

    # 2. If start_epoch > 0, look for the checkpoint in the training
    # output directory
    elif start_epoch > 0:
        # Construct the path matching the training save logic:
        # {prefix}tuned_models/epoch-{epoch}/{name}/...

        # Path A: Correct path based on finetune() save_dir logic
        trained_path = (
            f"{prefix}tuned_models/epoch-{start_epoch}"
            f"/{model_name}{quant_status}"
            f"/lora_numpert{rank_postfix}"
        )

        if os.path.exists(trained_path):
            print("📂 Mode: Found training checkpoint (Standard)")
            model_name_arg = trained_path
        else:
            print(
                "⚠️ Warning: Checkpoint for epoch "
                f"{start_epoch} not found locally."
            )
            print(f"   Checked: {trained_path}")
            print(
                f"   Attempting to load '{model_name}'" " from Hugging Face..."
            )
            model_name_arg = get_model_name_arg(model_name)

    # 3. Default to Hugging Face Hub or mapped alias
    else:
        print("☁️ Mode: Hugging Face / Alias")
        model_name_arg = get_model_name_arg(model_name)

    print(f"Loading model from: {model_name_arg}\n")

    # Load model
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
            f"⚠️  WARNING: {too_long}/{len(test_data)} examples"
            " exceed max_seq_length!"
        )
    else:
        print("✅ All examples fit within max_seq_length")

    # Prepare for predictions
    predictions = []  # Will contain None for parse failures
    raw_outputs = []  # Store raw outputs for debugging

    # Determine output file prefix for unperturbed-claims mode
    file_prefix = "unperturbed_" if inference_unperturbed_claims else ""

    # Create output directory
    pred_dir_name = build_output_dir_name(
        model_name,
        rank,
        start_epoch,
        load_in_4bit=load_in_4bit,
    )
    pred_dir = f"{output_dir}/{pred_dir_name}"
    os.makedirs(pred_dir, exist_ok=True)

    # Export inference configuration to JSON
    inference_config = build_inference_config(
        model_name=model_name,
        model_path=model_name_arg,
        test_file_configs=test_file_configs,
        test_dataset=test_data,
        rank=rank,
        start_epoch=start_epoch,
        expand_perturbations=expand_perturbations,
        max_seq_length=max_seq_length,
        timestamp=time_start,
    )
    config_path = save_inference_config(
        inference_config,
        pred_dir,
        filename=f"{file_prefix}inference_config.json",
    )
    print(f"Inference config saved to: {config_path}")

    # Run inference
    print("\nRunning inference...")

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

        label = extract_label_from_response(decoded)
        predictions.append(label)

        entry = {
            "index": i,
            "claim_id": test_data[i]["claim_id"],
            "perturbation_type": test_data[i]["perturbation_type"],
            "label": test_data[i]["label"],
            "full_prompt": prompt_text,  # Full input
            "raw_response": decoded,  # Model output only
            "parsed_successfully": label is not None,
            "predicted_label": label,
        }
        if "source_file" in test_data.column_names:
            entry["source_file"] = test_data[i]["source_file"]
        raw_outputs.append(entry)

    raw_outputs_path = f"{pred_dir}/{file_prefix}inference_details.json"
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs, f, indent=2)
    print(
        f"Inference details (prompts + outputs) saved to: {raw_outputs_path}\n"
    )

    # Build structured result records and compute metrics
    result_records = build_result_records(test_data, raw_outputs)
    report = compute_metrics(result_records)
    print_metrics_report(report, show_classification_reports=True)

    # Save all outputs
    metrics_path = save_metrics_report(
        report, pred_dir, filename=f"{file_prefix}metrics.json"
    )
    print(f"Metrics saved to: {metrics_path}")

    failures_path = save_parse_failures(
        result_records,
        raw_outputs,
        pred_dir,
        filename=f"{file_prefix}parse_failures.json",
    )
    if failures_path:
        print(f"Parse failure details saved to: {failures_path}")

    csv_path = save_predictions_csv(
        result_records,
        raw_outputs,
        pred_dir,
        filename=f"{file_prefix}predictions.csv",
    )
    print(f"Predictions saved to: {csv_path}\n")


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand perturbations in dataset",
    )
    parser.add_argument(
        "--inference_unperturbed_claims",
        action="store_true",
        default=False,
        help=(
            "Run inference using only normalized_claim "
            "(no perturbations). Outputs are prefixed with 'unperturbed_'."
        ),
    )

    # Inference options
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for predictions "
        "(auto-generates run name if not set)",
    )

    args = parser.parse_args()

    # Parse file configs for test data
    test_configs = [parse_file_config(p) for p in args.test_json]

    inference(
        test_file_configs=test_configs,
        model_name=args.model,
        rank=args.rank,
        start_epoch=args.start_epoch,
        expand_perturbations=args.expand_perturbations,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        inference_unperturbed_claims=args.inference_unperturbed_claims,
    )
