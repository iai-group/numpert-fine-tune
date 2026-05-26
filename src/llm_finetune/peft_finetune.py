"""PEFT-based fine-tuning for fact-checking models with numerical perturbations.

This module supports multiple parameter-efficient fine-tuning approaches:
- PromptTuning: Learn soft prompt prefixes
- PrefixTuning: Learn prefix at each transformer layer
- LoRA: Low-Rank Adaptation (via PEFT for consistency)
- Combined: LoRA + Prompt/Prefix Tuning together

Reuses data loading, tokenization, and training infrastructure from
unsloth_finetune.
"""

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Union, Dict, Any, cast
from enum import Enum

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    LoraConfig,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from src.llm_finetune.config import prefix
from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.llm_finetune.utils.data_loading.dataset_balancer import (
    MethodBalanceMode,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    parse_file_config,
)
from src.llm_finetune.utils.naming import generate_run_name
from src.llm_finetune.utils.training.training_config_export import (
    build_training_config,
    save_training_config,
)
from src.llm_finetune.utils.training.dataset_distribution_report import (
    DISTRIBUTION_REPORT_FILENAME,
    compute_distribution,
    format_distribution_report,
    save_distribution_report,
)


class PeftMethod(str, Enum):
    """Supported PEFT fine-tuning methods."""

    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    LORA = "lora"
    LORA_PROMPT = "lora_prompt"
    LORA_PREFIX = "lora_prefix"


def resolve_step_interval(
    step_value: Union[int, float],
    total_steps: Optional[int] = None,
    step_name: str = "steps",
) -> int:
    """Resolve a step interval from absolute or fractional CLI input.

    Fractional values in ``(0, 1]`` are multiplied by ``total_steps`` and
    ceiled so that the result is always **at least** the intended fraction of
    training.
    """
    if isinstance(step_value, float) and 0.0 < step_value <= 1.0:
        if total_steps is None:
            raise ValueError(
                f"total_steps is required when {step_name} is a fraction"
            )
        return max(1, math.ceil(total_steps * step_value))

    if isinstance(step_value, float):
        if step_value <= 0:
            raise ValueError(f"{step_name} must be positive, got {step_value}")
        return int(step_value)

    if step_value <= 0:
        raise ValueError(f"{step_name} must be positive, got {step_value}")

    return step_value


def get_model_name_arg(model_name: str) -> str:
    """Convert short model name to full HuggingFace path."""
    if model_name.startswith("Qwen3-8B") or model_name.startswith(
        "unsloth/Qwen3-8B"
    ):
        return "Qwen/Qwen2.5-8B-Instruct"
    elif model_name.startswith("Qwen3-32B") or model_name.startswith(
        "unsloth/Qwen3-32B"
    ):
        return "Qwen/Qwen2.5-32B-Instruct"
    elif model_name.startswith("llama-3.2-3b"):
        return "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b"):
        return "meta-llama/Llama-3.2-1B-Instruct"
    return model_name


def setup_chat_template(tokenizer, model_name: str):
    """Set chat template for model based on model name."""
    if "llama" in model_name.lower():
        # Llama-3.2 and Llama-3.1 use ChatML
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set ns = namespace(offset=1) %}"
            "{%- endif %}"
            "{%- for message in messages[ns.offset:] %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}"
            "<|im_end|>\n"
            "{%- endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
    elif "qwen" in model_name.lower():
        # Qwen uses ChatML
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set ns = namespace(offset=1) %}"
            "{%- else %}"
            "{% set ns = namespace(offset=0) %}"
            "{%- endif %}"
            "{%- for message in messages[ns.offset:] %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}"
            "<|im_end|>\n"
            "{%- endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )

    # Set padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def create_peft_model(
    model,
    peft_method: PeftMethod,
    rank: int = 32,
    lora_alpha: Optional[int] = None,
    num_virtual_tokens: Optional[int] = None,
    num_prefix_tokens: Optional[int] = None,
) -> Any:
    """Create PEFT model with specified method.

    Args:
        model: Base transformer model
        peft_method: Which PEFT method to apply
        rank: LoRA rank (used for LoRA methods)
        lora_alpha: LoRA alpha parameter
        num_virtual_tokens: Number of virtual tokens for prompt tuning
        num_prefix_tokens: Number of prefix tokens for prefix tuning

    Returns:
        PEFT-wrapped model
    """
    if lora_alpha is None:
        lora_alpha = rank // 2

    if peft_method == PeftMethod.LORA:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    elif peft_method == PeftMethod.PROMPT_TUNING:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens or 20,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            prompt_tuning_init_text=None,  # Use random initialization
        )

    elif peft_method == PeftMethod.PREFIX_TUNING:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_prefix_tokens or 20,
            prefix_projection=True,
            encoder_hidden_size=None,  # Auto-detect from model
        )

    elif peft_method == PeftMethod.LORA_PROMPT:
        # Combine LoRA + Prompt Tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            random_state=3407,
        )

        # First apply LoRA
        model = get_peft_model(model, lora_config)

        # Then apply Prompt Tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens or 20,
            prompt_tuning_init=PromptTuningInit.RANDOM,
        )
        return get_peft_model(model, peft_config)

    elif peft_method == PeftMethod.LORA_PREFIX:
        # Combine LoRA + Prefix Tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            random_state=3407,
        )

        # First apply LoRA
        model = get_peft_model(model, lora_config)

        # Then apply Prefix Tuning
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_prefix_tokens or 20,
            prefix_projection=True,
        )
        return get_peft_model(model, peft_config)

    else:
        raise ValueError(f"Unknown PEFT method: {peft_method}")

    return get_peft_model(model, peft_config)


def count_trainable_params(model) -> tuple:
    """Count total and trainable parameters.

    Returns:
        (total_params, trainable_params, trainable_percentage)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    trainable_pct = 100.0 * trainable_params / total_params

    return total_params, trainable_params, trainable_pct


class EfficiencyTracker(TrainerCallback):
    """Callback to track training and inference efficiency metrics."""

    def __init__(self) -> None:
        """Initialize efficiency tracker."""
        self.epoch_start_time: Optional[float] = None
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self.batch_start_time: Optional[float] = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Record epoch start time."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Record epoch end time."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

    def on_step_begin(self, args, state, control, **kwargs):
        """Record batch start time."""
        self.batch_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Record batch end time."""
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)

    def get_stats(self) -> Dict[str, float]:
        """Get efficiency statistics."""
        if not self.epoch_times:
            return {}

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        avg_batch_time = (
            (sum(self.batch_times) / len(self.batch_times))
            if self.batch_times
            else 0
        )

        return {
            "avg_epoch_time_seconds": avg_epoch_time,
            "avg_batch_time_seconds": avg_batch_time,
            "total_training_time_seconds": sum(self.epoch_times),
        }


def finetune_peft(  # noqa: C901
    train_json_paths: Optional[List[str]] = None,
    val_json_paths: Optional[List[str]] = None,
    train_file_configs: Optional[List[FileLoadConfig]] = None,
    val_file_configs: Optional[List[FileLoadConfig]] = None,
    model_name: str = "Qwen/Qwen2.5-8B-Instruct",
    perturb_type_names: str = "",
    peft_method: PeftMethod = PeftMethod.LORA,
    rank: int = 32,
    num_virtual_tokens: Optional[int] = None,
    num_prefix_tokens: Optional[int] = None,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 16384,
    use_perturbations: bool = True,
    balance_labels: bool = False,
    balance_labels_mode: Literal["undersample", "oversample"] = "undersample",
    perturb_ratio: Optional[float] = None,
    false_perturb_t_to_f_ratio: Optional[float] = None,
    method_balance_mode: Optional[MethodBalanceMode] = None,
    eval_steps: Union[int, float] = 500,
    save_steps: Optional[int] = None,
    sample_percent: float = 100,
    start_epoch: int = 0,
    save_gguf: bool = False,
    gguf_quantization_method: str = "q4_k_m",
    debug: bool = False,
):
    """Fine-tune model using PEFT methods for fact-checking.

    Uses parameter-efficient fine-tuning methods for numerical perturbations.
    """
    time_start = datetime.now()

    print("\n" + "=" * 80)
    print("STARTING PEFT FINE-TUNING")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"PEFT Method: {peft_method.value}")
    print(f"Rank (LoRA): {rank}")
    if num_virtual_tokens:
        print(f"Virtual Tokens (Prompt/Prefix): {num_virtual_tokens}")
    print(f"Number of training epochs: {num_train_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Use perturbations: {use_perturbations}")
    print(f"Balance labels: {balance_labels}")
    print("=" * 80 + "\n")

    # Load base model (without 4-bit quantization for PEFT compatibility)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Setup chat template
    setup_chat_template(tokenizer, model_name)

    # Load and prepare data (reuse from unsloth_finetune)
    print("\nLoading training data...")
    train_data = load_split(
        file_configs=train_file_configs,
        split="train",
        use_perturbations=use_perturbations,
        include_doc=True,
        balance_labels=balance_labels,
        balance_labels_mode=balance_labels_mode,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        method_balance_mode=method_balance_mode,
        seed=3407 + start_epoch,
    )
    print(f"Loaded {len(train_data)} training examples")

    # Apply data sampling if requested
    if sample_percent < 100:
        original_size = len(train_data)
        sample_size = max(1, int(len(train_data) * sample_percent / 100))
        train_data = train_data.select(
            list(range(0, original_size, max(1, original_size // sample_size)))[
                :sample_size
            ]
        )
        print(
            f"Sampled {len(train_data)} training examples ({sample_percent}% of {original_size})"  # noqa: E501
        )

    print("Loading validation data...")
    val_data = load_split(
        file_configs=val_file_configs,
        split="validation",
        use_perturbations=use_perturbations,
        include_doc=True,
        balance_labels=False,  # Don't balance validation set
        perturb_ratio=None,
        seed=3407 + start_epoch,
    )
    print(f"Loaded {len(val_data)} validation examples")

    # Compute and log distribution reports
    train_dist_stats = compute_distribution(train_data)
    val_dist_stats = compute_distribution(val_data)
    print(
        format_distribution_report(
            train_dist_stats, "TRAIN DATASET DISTRIBUTION"
        )
    )
    print(
        format_distribution_report(
            val_dist_stats, "VALIDATION DATASET DISTRIBUTION"
        )
    )

    # Tokenization function (reuse from unsloth but simplified)
    def tokenize_func(examples):
        """Tokenize conversations for training."""
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for conv in examples["conversations"]:
            if len(conv) < 2 or conv[-1]["role"] != "assistant":
                print(f"⚠️ Warning: Unexpected conversation format: {conv}")
                continue

            # Format full text with chat template
            full_text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Format prompt only (for masking)
            prompt_text = tokenizer.apply_chat_template(
                conv[:-1], tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=False,
            )
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=False,
            )

            # Create labels with masking
            labels = full_tokens["input_ids"].copy()
            prompt_len = len(prompt_tokens["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

            all_input_ids.append(full_tokens["input_ids"])
            all_labels.append(labels)
            all_attention_mask.append(full_tokens["attention_mask"])

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }

    # Apply tokenization
    print("Tokenizing datasets...")
    train_data = train_data.map(
        tokenize_func,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing training data",
    )
    val_data = val_data.map(
        tokenize_func,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizing validation data",
    )

    # Generate run name and output directory
    models_dir = f"{prefix}/models"
    run_name = generate_run_name(
        model_name=model_name,
        perturb_type_names=perturb_type_names,
        base_path=models_dir,
    )
    base_dir = f"{models_dir}/{run_name}_{peft_method.value}"
    output_dir = f"{base_dir}/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    # Create PEFT model
    print(f"\nApplying PEFT method: {peft_method.value}")
    model = create_peft_model(
        model,
        peft_method=peft_method,
        rank=rank,
        num_virtual_tokens=num_virtual_tokens,
        num_prefix_tokens=num_prefix_tokens,
    )

    # Count parameters
    total_params, trainable_params, trainable_pct = count_trainable_params(
        model
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_pct:.2f}%")

    # Export training configuration
    training_config_dict = build_training_config(
        model_name=model_name,
        train_file_configs=train_file_configs,
        val_file_configs=val_file_configs,
        train_dataset=train_data,
        val_dataset=val_data,
        rank=rank,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        use_perturbations=use_perturbations,
        balance_labels=balance_labels,
        balance_labels_mode=balance_labels_mode,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        method_balance_mode=method_balance_mode,
        timestamp=time_start,
    )
    training_config_dict["peft_method"] = peft_method.value
    training_config_dict["total_parameters"] = total_params
    training_config_dict["trainable_parameters"] = trainable_params
    training_config_dict["trainable_percentage"] = trainable_pct

    config_path = save_training_config(training_config_dict, base_dir)
    print(f"Training config saved to: {config_path}")

    # Save distribution report
    dist_report_path = save_distribution_report(
        train_dist_stats,
        val_dist_stats,
        Path(base_dir) / DISTRIBUTION_REPORT_FILENAME,
    )
    print(f"Distribution report saved to: {dist_report_path}")

    # Setup wandb
    wandb.init(  # type: ignore
        project="numerical_perturbfinetune_peft",
        name=(
            f"{model_name.split('/')[-1]}_{peft_method.value}"
            f"_epoch_{num_train_epochs}"
        ),
        config={
            "model": model_name,
            "peft_method": peft_method.value,
            "rank": rank,
            "num_virtual_tokens": num_virtual_tokens,
            "num_prefix_tokens": num_prefix_tokens,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "use_perturbations": use_perturbations,
            "balance_labels": balance_labels,
            "perturb_ratio": perturb_ratio,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_pct,
        },
    )

    # Calculate eval steps
    total_steps = None
    if isinstance(eval_steps, float) and 0.0 < eval_steps <= 1.0:
        steps_per_epoch = math.ceil(
            len(train_data) / (batch_size * gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * num_train_epochs
        resolved_eval_steps = resolve_step_interval(
            eval_steps,
            total_steps=total_steps,
            step_name="eval_steps",
        )
        eval_steps = resolved_eval_steps
    else:
        eval_steps = resolve_step_interval(eval_steps, step_name="eval_steps")

    # Setup training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps or eval_steps,
        save_total_limit=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_dir=f"{base_dir}/logs",
        report_to=["wandb"],
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
    )

    # Create efficiency tracker
    efficiency_tracker = EfficiencyTracker()

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        callbacks=[efficiency_tracker],
    )

    # Train
    print("\n" + "=" * 80)
    print("TRAINING STARTING")
    print("=" * 80 + "\n")

    train_result = trainer.train()

    # Log efficiency metrics
    efficiency_stats = efficiency_tracker.get_stats()
    for metric_name, value in efficiency_stats.items():
        wandb.log({f"efficiency/{metric_name}": value})  # type: ignore

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Training loss: {train_result.training_loss:.4f}")

    # Save model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    print(f"Model saved to: {output_dir}")

    # Merge LoRA adapter and export to GGUF (LoRA only — prompt/prefix tuning
    # have no weight changes to merge so GGUF would equal the base model)
    if save_gguf and peft_method in (
        PeftMethod.LORA,
        PeftMethod.LORA_PROMPT,
        PeftMethod.LORA_PREFIX,
    ):
        print("\nMerging LoRA adapter into base model...")
        from src.llm_finetune.utils.training.convert_to_gguf import (
            convert_existing_model,
        )

        merged_dir = f"{base_dir}/merged"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

        print(f"\nConverting to GGUF ({gguf_quantization_method})...")
        gguf_dir = f"{base_dir}/gguf_{gguf_quantization_method}"
        convert_existing_model(
            model_path=merged_dir,
            output_dir=gguf_dir,
            quantization_method=gguf_quantization_method,
        )
        print(f"GGUF saved to: {gguf_dir}")

    # Finish wandb run
    wandb.finish()  # type: ignore

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PEFT-based fine-tuning for fact-checking models"
    )

    # Required arguments
    parser.add_argument(
        "--train_json",
        type=str,
        nargs="+",
        required=True,
        help="Training JSON file path(s), supports globs and type filtering",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        nargs="+",
        required=True,
        help="Validation JSON file path(s), supports globs and type filtering",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Model name or path",
    )

    # PEFT method arguments
    parser.add_argument(
        "--peft_method",
        type=str,
        choices=[m.value for m in PeftMethod],
        default=PeftMethod.LORA.value,
        help="PEFT method to use",
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
        help="Number of virtual tokens for prompt/prefix tuning",
    )

    # Training arguments
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch (for resume or tracking)",
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
        help="Batch size per device",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Maximum sequence length",
    )

    # Data arguments
    parser.add_argument(
        "--use_perturbations",
        action="store_true",
        default=True,
        help="Use perturbed examples in training",
    )
    parser.add_argument(
        "--no-use_perturbations",
        dest="use_perturbations",
        action="store_false",
        help="Disable perturbed examples in training",
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        default=False,
        help="Balance true/false label distribution",
    )
    parser.add_argument(
        "--balance_labels_mode",
        type=str,
        default="oversample",
        choices=["undersample", "oversample"],
        help="How to balance labels: oversample or undersample.",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=None,
        help="Target ratio of perturbed examples",
    )
    parser.add_argument(
        "--false_perturb_t_to_f_ratio",
        type=float,
        default=None,
        help="Ratio of False to True in perturbed data",
    )
    parser.add_argument(
        "--method_balance_mode",
        type=str,
        default=None,
        help="Method for balancing (oversample, undersample, etc.)",
    )
    parser.add_argument(
        "--perturb_type_names",
        type=str,
        default=None,
        help="Perturbation types to use (comma-separated)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_steps",
        type=float,
        default=500,
        help="Eval frequency (int=steps, float in (0,1]=fraction)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save frequency (default=eval_steps)",
    )
    parser.add_argument(
        "--sample_percent",
        type=float,
        default=100,
        help="Percentage of training data to use (e.g., 10 for 10% sample)",
    )
    parser.add_argument(
        "--save_gguf",
        action="store_true",
        default=False,
        help="Merge LoRA adapter and export to GGUF after training",
    )
    parser.add_argument(
        "--gguf_quantization_method",
        type=str,
        default="q4_k_m",
        help="GGUF quantization method (default: q4_k_m)",
    )

    args = parser.parse_args()

    # Parse file configurations (supports path:type1,type2 notation)
    train_file_configs = [parse_file_config(p) for p in args.train_json]
    val_file_configs = [parse_file_config(p) for p in args.val_json]

    finetune_peft(
        train_file_configs=train_file_configs,
        val_file_configs=val_file_configs,
        model_name=args.model,
        peft_method=PeftMethod(args.peft_method),
        rank=args.rank,
        num_virtual_tokens=args.num_virtual_tokens,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        use_perturbations=args.use_perturbations,
        balance_labels=args.balance_labels,
        balance_labels_mode=cast(
            Literal["undersample", "oversample"], args.balance_labels_mode
        ),
        perturb_ratio=args.perturb_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        sample_percent=args.sample_percent,
        save_gguf=args.save_gguf,
        gguf_quantization_method=args.gguf_quantization_method,
    )
