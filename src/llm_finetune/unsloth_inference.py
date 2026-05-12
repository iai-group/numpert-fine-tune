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
from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.numpert_prompt.prompt import PromptType
from src.llm_finetune.utils.data_loading.perturbation_types import (
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
from src.llm_finetune.utils.inference.resume import load_resume_data
from src.llm_finetune.utils.inference.unperturbed_claims import (
    deduplicate_unperturbed_claims,
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


_CLI_PROMPT_TYPE_MAP = {
    "ZERO_SHOT": PromptType.ZERO_SHOT,
    "FEW_SHOT": PromptType.FEW_SHOT,
    "NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING": (
        PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING
    ),
}


def inference(  # noqa: C901
    test_json_paths: Optional[List[str]] = None,
    test_file_configs: Optional[List[FileLoadConfig]] = None,
    model_name: str = "unsloth/Qwen3-8B",
    rank: int = 32,
    start_epoch: int = 0,
    use_perturbations: bool = True,
    max_seq_length: int = 16384,
    output_dir: Optional[str] = None,
    inference_unperturbed_claims: bool = False,
    tokenizer_name_or_path: Optional[str] = None,
    verbose: bool = False,
    require_gpu_offload: bool = False,
    resume: bool = False,
    prompt_type: PromptType = PromptType.ZERO_SHOT,
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
        use_perturbations = False

    print("\n" + "=" * 80)
    print(
        "STARTING INFERENCE"
        + (" (UNPERTURBED CLAIMS)" if inference_unperturbed_claims else "")
    )
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model from Epoch: {start_epoch}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Using perturbations: {use_perturbations}")
    cli_prompt_type = next(
        k for k, v in _CLI_PROMPT_TYPE_MAP.items() if v is prompt_type
    )
    print(f"Prompt type: {cli_prompt_type}")
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

    # 1. Check if the user provided a direct path to a local GGUF file
    if os.path.isfile(model_name) and model_name.endswith(".gguf"):
        print("🦙 Mode: Direct local GGUF file")
        model_name_arg = model_name

    # 2. Check if the user provided a direct path to a local folder
    elif os.path.isdir(model_name):
        print("📂 Mode: Direct local folder")
        model_name_arg = model_name

    # 3. If start_epoch > 0, look for the checkpoint in the training
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

    # 4. Default to Hugging Face Hub or mapped alias
    else:
        print("☁️ Mode: Hugging Face / Alias")
        model_name_arg = get_model_name_arg(model_name)

    print(f"Loading model from: {model_name_arg}\n")

    is_gguf = model_name_arg.endswith(".gguf")

    # Load model
    if is_gguf:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with CUDA support (pre-built wheel):\n"
                "  pip install llama-cpp-python "
                "--extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ) from e
        try:
            from unsloth.chat_templates import get_chat_template
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "unsloth.chat_templates is required for GGUF/chat-template "
                "inference but is not available in this environment."
            ) from e
        print("🦙 Mode: GGUF via llama-cpp-python")

        # Load the HF tokenizer so we can apply the exact same chat
        # template that was used during fine-tuning.
        if tokenizer_name_or_path is None:
            raise ValueError(
                "--tokenizer is required when loading a GGUF model. "
                "Pass the HuggingFace model name or path whose tokenizer "
                "was used during fine-tuning (e.g. unsloth/Qwen3-8B)."
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if "llama-3" in tokenizer_name_or_path.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
            # <|eot_id|> ends each turn; <|end_of_text|> is hard EOS
            gguf_stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
        else:  # Qwen / ChatML
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")
            gguf_stop_tokens = ["<|im_end|>", "<|endoftext|>"]

        model = Llama(
            model_path=model_name_arg,
            n_ctx=max_seq_length,
            n_gpu_layers=-1,
            n_batch=max_seq_length,
            # flash_attn=True,
            # The chat template already contains BOS; prevent llama.cpp
            # from prepending a second one.
            add_bos_token=False,
            verbose=verbose,
        )

        # Verify GPU offloading is active
        import llama_cpp

        _cuda_install_hint = (
            "pip install llama-cpp-python "
            "--extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu124 "
            "--force-reinstall --no-cache-dir"
        )
        if not hasattr(llama_cpp, "llama_supports_gpu_offload"):
            print(
                "⚠️  WARNING: Cannot verify GPU offload support. "
                "If inference is slow, reinstall with CUDA:\n"
                f"   {_cuda_install_hint}"
            )
        elif not llama_cpp.llama_supports_gpu_offload():
            _msg = (
                "llama-cpp-python was built WITHOUT GPU support. "
                "All layers are running on CPU. Reinstall with CUDA:\n"
                f"  {_cuda_install_hint}"
            )
            if require_gpu_offload:
                raise RuntimeError(_msg)
            else:
                print(f"⚠️  WARNING: {_msg}")
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
    test_data = load_split(
        json_paths=test_json_paths,
        file_configs=test_file_configs,
        split="test",
        use_perturbations=use_perturbations,
        include_doc=True,
        prompt_type=prompt_type,
    )

    if inference_unperturbed_claims:
        pre_dedup = len(test_data)
        test_data = deduplicate_unperturbed_claims(test_data)
        print(
            f"Unperturbed dedup: {pre_dedup} → {len(test_data)} "
            f"(removed {pre_dedup - len(test_data)} duplicates)"
        )

    print(f"Test dataset size: {len(test_data)}")

    # Check sequence lengths
    print("\nChecking sequence lengths...")
    too_long = 0
    max_length_seen = 0

    for i, example in enumerate(test_data):
        if is_gguf:
            # Format via the same template used in training, then count
            # tokens with llama.cpp's own tokenizer (add_bos=False because
            # add_bos_token=False is set on the Llama instance).
            formatted = tokenizer.apply_chat_template(
                example["conversations"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            token_count = len(
                model.tokenize(formatted.encode("utf-8"), add_bos=False)
            )
        else:
            formatted = tokenizer.apply_chat_template(
                example["conversations"][:-1],
                tokenize=False,
                add_generation_prompt=True,
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

    raw_outputs: list[dict] = [{} for _ in range(len(test_data))]

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
        use_perturbations=use_perturbations,
        max_seq_length=max_seq_length,
        timestamp=time_start,
    )
    config_path = save_inference_config(
        inference_config,
        pred_dir,
        filename=f"{file_prefix}inference_config.json",
    )
    print(f"Inference config saved to: {config_path}")

    # Deduplicate: only evaluate each original_normalized claim once.
    # Perturbed entries are always unique across files (different claim text)
    # so only original_normalized entries can be true duplicates.
    seen_original_claims: dict[str, int] = {}  # claim_id → first index
    unique_indices: list[int] = []
    duplicate_map: dict[int, int] = {}  # dup index → first occurrence index

    for i in range(len(test_data)):
        if test_data[i]["perturbation_type"] == "original_normalized":
            cid = test_data[i]["claim_id"]
            if cid not in seen_original_claims:
                seen_original_claims[cid] = i
                unique_indices.append(i)
            else:
                duplicate_map[i] = seen_original_claims[cid]
        else:
            unique_indices.append(i)

    n_duplicates = len(duplicate_map)
    if n_duplicates > 0:
        print(
            f"\nDeduplication: {len(test_data)} entries → "
            f"{len(unique_indices)} unique evaluations "
            f"(skipping {n_duplicates} duplicate original_normalized claims)"
        )

    # Resume support: load previously completed results if the file exists
    raw_outputs_path = f"{pred_dir}/{file_prefix}inference_details.jsonl"
    completed_indices: set[int] = set()
    if resume and os.path.exists(raw_outputs_path):
        raw_outputs, completed_indices = load_resume_data(
            raw_outputs_path, len(raw_outputs)
        )
        if completed_indices:
            print(
                f"\nResuming: loaded {len(completed_indices)} "
                f"previously completed results from {raw_outputs_path}"
            )

    # Filter out already-completed indices
    remaining_indices = [
        i for i in unique_indices if i not in completed_indices
    ]

    # Run inference on remaining entries only
    print(
        f"\nRunning inference on {len(remaining_indices)}/{len(unique_indices)}"
        " unique entries..."
    )

    # When resuming JSONL writes, ensure the file ends with a newline so
    # the next appended record always starts on its own line.
    if (
        resume
        and os.path.exists(raw_outputs_path)
        and os.path.getsize(raw_outputs_path) > 0
    ):
        with open(raw_outputs_path, "rb+") as f_fix:
            f_fix.seek(-1, os.SEEK_END)
            if f_fix.read(1) != b"\n":
                f_fix.write(b"\n")

    write_mode = "a" if resume else "w"
    with open(raw_outputs_path, write_mode, encoding="utf-8") as f_out:
        for i in tqdm(remaining_indices):
            messages = test_data[i]["conversations"][:-1]

            if is_gguf:
                # Format the prompt with the same chat template used in
                # training. Use raw text completion (not
                # create_chat_completion) so llama.cpp doesn't apply its
                # own built-in template on top.
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                response = model(
                    prompt_text,
                    max_tokens=8192,
                    temperature=0.001,
                    stop=gguf_stop_tokens,
                )
                decoded = response["choices"][0]["text"]
            else:
                # HuggingFace/unsloth: apply chat template and generate
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = tokenizer.apply_chat_template(
                    messages,
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
                    max_new_tokens=8192,
                    use_cache=True,
                    temperature=0.001,
                )

                # Decode only new tokens
                new_tokens = outputs[0][input_length:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

            label = extract_label_from_response(decoded)

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
            raw_outputs[i] = entry
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

        # Fill duplicate entries by reusing the inference result
        remaining_dups = {
            k: v for k, v in duplicate_map.items() if k not in completed_indices
        }
        for dup_idx, orig_idx in remaining_dups.items():
            orig = raw_outputs[orig_idx]
            entry = {
                "index": dup_idx,
                "claim_id": test_data[dup_idx]["claim_id"],
                "perturbation_type": test_data[dup_idx]["perturbation_type"],
                "label": test_data[dup_idx]["label"],
                "full_prompt": orig["full_prompt"],
                "raw_response": orig["raw_response"],
                "parsed_successfully": orig["parsed_successfully"],
                "predicted_label": orig["predicted_label"],
                "reused_from_index": orig_idx,
            }
            if "source_file" in test_data.column_names:
                entry["source_file"] = test_data[dup_idx]["source_file"]
            raw_outputs[dup_idx] = entry
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

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


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the inference CLI."""
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
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "HuggingFace tokenizer name or path. Required when --model is a "
            "GGUF file. Must match the tokenizer used during fine-tuning "
            "(e.g. unsloth/Qwen3-8B) so the correct chat template is applied."
        ),
    )

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
        "--use_perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include perturbed examples in the dataset. "
            "Pass --no-use_perturbations to run inference on original "
            "(normalised) claims only."
        ),
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose llama.cpp output (CUDA graph warmup, perf stats).",
    )
    parser.add_argument(
        "--require-gpu-offload",
        action="store_true",
        default=False,
        help="Fail hard if llama-cpp-python was built "
        "without GPU offload support. "
        "By default, a warning is printed and CPU inference proceeds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume a previous inference run by loading already-completed "
            "results from an existing inference_details.jsonl in the output "
            "directory. Without this flag, inference always starts from "
            "scratch even if a prior output file exists."
        ),
    )
    parser.add_argument(
        "--prompt_type",
        default="ZERO_SHOT",
        choices=list(_CLI_PROMPT_TYPE_MAP.keys()),
        help=(
            "User prompt type to use during inference. "
            "ZERO_SHOT (default): no examples. "
            "FEW_SHOT: two-shot true/false examples. "
            "NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING: synthetic "
            "numerical perturbation examples with reasoning chains."
        ),
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Parse file configs for test data
    test_configs = [parse_file_config(p) for p in args.test_json]

    inference(
        test_file_configs=test_configs,
        model_name=args.model,
        rank=args.rank,
        start_epoch=args.start_epoch,
        use_perturbations=args.use_perturbations,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        inference_unperturbed_claims=args.inference_unperturbed_claims,
        tokenizer_name_or_path=args.tokenizer,
        verbose=args.verbose,
        require_gpu_offload=args.require_gpu_offload,
        resume=args.resume,
        prompt_type=_CLI_PROMPT_TYPE_MAP[args.prompt_type],
    )
