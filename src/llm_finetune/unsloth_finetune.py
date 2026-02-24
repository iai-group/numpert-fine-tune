"""Fine-tuning and inference for fact-checking models with numerical
perturbations."""

import re
import json
import src
import json
from typing import Optional, List, Dict
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime

from src.llm_finetune.config import prefix
from src.llm_finetune.hf_dataset_instructions import (
    load_fact_checking_data,
    load_train_and_validation,
    PerturbationType,
)
from src.llm_finetune.class_balancer import DatasetBalancer
from src.llm_finetune.batch_sampler import EnhancedPerturbationBatchSampler


from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    AutoTokenizer,
)
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import wandb
import argparse
import os
from tqdm import tqdm
import pandas as pd

# Valid labels for fact-checking (JSON format)
VALID_LABELS = ['{"label": true}', '{"label": false}']

# Timestamp for model naming
time_start = datetime.now()
model_name_timestamp = time_start.strftime("%Y%m%d_%H%M%S")


class PerturbationAwareSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that uses EnhancedPerturbationBatchSampler for
    training.

    This allows claim_id-aware batching where examples from the same claim are
    grouped together to help the model learn contrasts between original and
    perturbed versions.
    """

    def __init__(
        self,
        *args,
        use_custom_sampler: bool = False,
        pre_built_batch_sampler=None,  # Accept pre-built sampler
        **kwargs,
    ):
        """
        Args:
            use_custom_sampler: Whether to use EnhancedPerturbationBatchSampler
            pre_built_batch_sampler: Pre-built batch sampler (created before columns removed)
        """
        super().__init__(*args, **kwargs)

        self.use_custom_sampler = use_custom_sampler
        self.pre_built_batch_sampler = pre_built_batch_sampler

    def get_train_dataloader(self) -> DataLoader:
        """Override to use custom batch sampler when enabled."""
        if not self.use_custom_sampler or self.pre_built_batch_sampler is None:
            # Use default SFTTrainer data loading
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        print("\n" + "=" * 80)
        print("🔧 Using EnhancedPerturbationBatchSampler for training")
        print("=" * 80)

        # Use the pre-built batch sampler
        batch_sampler = self.pre_built_batch_sampler

        # Create DataLoader with custom batch sampler
        # Note: When using batch_sampler, we must NOT specify batch_size
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        print(f"✅ Custom dataloader created with {len(batch_sampler)} batches")
        print("=" * 80 + "\n")

        return dataloader


class ValidationCallback(TrainerCallback):
    """Callback to run validation and print metrics during training."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print("\n" + "=" * 80)
            print("VALIDATION METRICS:")
            print("=" * 80)
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("=" * 80 + "\n")


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


def finetune(
    train_json_paths: List[str],
    val_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    perturb_type_names: str = "",
    rank: int = 32,
    num_train_epochs: int = 1,
    start_epoch: int = 1,
    resume_model: Optional[str] = None,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 16384,
    expand_perturbations: bool = True,
    balance_labels: bool = False,
    perturb_ratio: float = 1.0,
    false_perturb_t_to_f_ratio: float = 0.67,
    batch_match_claim_id: bool = False,
    split_large_claims: bool = True,
    ensure_original_in_batch: bool = True,
    ensure_perturbed_in_batch: bool = True,
    eval_steps: int = 100,
    save_steps: int = 500,
    debug: bool = False,
):
    target_epoch = start_epoch + num_train_epochs - 1
    """Fine-tune a model for fact-checking with numerical perturbations."""

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Rank: {rank}")
    print(f"Resume epoch: {start_epoch}")
    print(f"Number of training epochs set: {num_train_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Expand perturbations: {expand_perturbations}")
    print(f"Balance labels: {balance_labels}")
    print(f"Perturb ratio: {perturb_ratio}")
    print(f"False perturb T→F ratio: {false_perturb_t_to_f_ratio}")
    print(f"Batch match claim_id: {batch_match_claim_id}")
    if batch_match_claim_id:
        print(f"  - Split large claims: {split_large_claims}")
        print(f"  - Ensure original in batch: {ensure_original_in_batch}")
        print(f"  - Ensure perturbed in batch: {ensure_perturbed_in_batch}")
    print("=" * 80 + "\n")

    # Model configuration
    dtype = None
    load_in_4bit = True

    # Naming conventions
    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"

    # Determine model path
    if start_epoch == 1:
        model_name_arg = get_model_name_arg(model_name)
    else:
        if resume_model is None:
            raise ValueError(
                f"--resume_model is required when start_epoch > 1. "
                f"Provide the path to a previously saved merged model."
            )
        model_name_arg = resume_model

    print(f"Loading model from: {model_name_arg}\n")

    # Load model (FIX: use model_name_arg instead of hardcoded value)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_arg,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        fix_tokenizer=False,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Set chat template
    if "llama-3" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    elif "qwen" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    else:  # TODO fix for other models
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # Set padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Formatting function with length monitoring
    truncated_count = 0
    total_count = 0

    def tokenize_func(examples):
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for conv in examples["conversations"]:
            if len(conv) < 2 or conv[-1]["role"] != "assistant":
                print(f"⚠️ Warning: Unexpected conversation format: {conv}")
                # Instead of continue, create a dummy example
                dummy_text = "Error"
                dummy_tokens = tokenizer(
                    dummy_text,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
                all_input_ids.append(dummy_tokens["input_ids"])
                all_labels.append([-100] * len(dummy_tokens["input_ids"]))
                all_attention_mask.append(dummy_tokens["attention_mask"])
                continue

            # 1. Format the full text (Instruction + Response + EOS)
            full_text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,  # Crucial: False includes the EOS token
            )

            # 2. Format the prompt only (Instruction + Generation Header)
            prompt_text = tokenizer.apply_chat_template(
                conv[:-1], tokenize=False, add_generation_prompt=True
            )

            # 3. Tokenize both
            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=False,  # Template already adds them
            )

            prompt_tokens = tokenizer(
                prompt_text, add_special_tokens=False, truncation=False
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            # 4. Mask the labels: -100 for prompt, actual ID for response
            labels = [-100] * len(input_ids)
            prompt_len = len(prompt_tokens["input_ids"])

            # Only unmask tokens from the end of the prompt to the end of the full sequence
            for i in range(prompt_len, len(input_ids)):
                labels[i] = input_ids[i]

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }

    print("Loading datasets...")
    datasets = load_train_and_validation(
        train_json_paths=train_json_paths,
        val_json_paths=val_json_paths,
        expand_perturbations_flag=expand_perturbations,
        include_doc=True,
    )

    train_data = datasets["train"]
    val_data = datasets["validation"]

    print(f"Loaded train dataset size: {len(train_data)}")
    print(f"Loaded validation dataset size: {len(val_data)}")

    # Apply balancing to training data
    if balance_labels or perturb_ratio != 1.0:
        balancer = DatasetBalancer(
            dataset=train_data,
            balance_labels=balance_labels,
            perturb_ratio=perturb_ratio,
            false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
            seed=3407 + start_epoch,
        )
        train_data = balancer.balance()

    print(f"\nFinal train dataset size: {len(train_data)}")

    # Show perturbation distribution
    train_perturb_dist = Counter(train_data["perturbation_type"])
    print("\nTrain perturbation distribution:")
    for p_type, count in train_perturb_dist.most_common():
        print(f"  {p_type:30s}: {count:5d} ({100*count/len(train_data):5.1f}%)")

    # Show label distribution
    train_label_dist = Counter(train_data["ground_truth"])
    print("\nTrain label distribution:")
    for label, count in train_label_dist.most_common():
        print(f"  {label:30s}: {count:5d} ({100*count/len(train_data):5.1f}%)")

    # ============================================================================
    # DEBUG: Check conversation format
    # ============================================================================
    if debug:
        print("\n" + "=" * 80)
        print("DEBUGGING CONVERSATION FORMAT")
        print("=" * 80)

        def debug_conversation(dataset, dataset_name, num_samples=3):
            print(f"\n{dataset_name} Dataset:")
            print(f"  Total examples: {len(dataset)}")
            print(f"  Columns: {dataset.column_names}")

            # Check conversation structure
            has_assistant = 0
            missing_assistant = 0

            for example in dataset:
                conv = example["conversations"]
                if len(conv) >= 2 and conv[-1]["role"] == "assistant":
                    has_assistant += 1
                else:
                    missing_assistant += 1

            print(f"\n  Conversations with assistant response: {has_assistant}")
            print(
                f"  Conversations WITHOUT assistant response: {missing_assistant}"
            )

            # Show sample conversations
            print(f"\n  Sample conversations (showing first {num_samples}):")
            for i in range(min(num_samples, len(dataset))):
                example = dataset[i]
                conv = example["conversations"]

                print(f"\n  --- Example {i} ---")
                print(f"  Claim ID: {example.get('claim_id', 'N/A')}")
                print(
                    f"  Perturbation: {example.get('perturbation_type', 'N/A')}"
                )
                print(f"  Ground truth: {example.get('ground_truth', 'N/A')}")
                print(f"  Conversation structure:")

                for j, msg in enumerate(conv):
                    role = msg["role"]
                    content = (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    )
                    print(f"    [{j}] {role}: {repr(content)}")

                # Check if last message is assistant
                if conv[-1]["role"] == "assistant":
                    print(f"  ✅ Last message is from assistant")
                else:
                    print(
                        f"  ❌ Last message is NOT from assistant (role: {conv[-1]['role']})"
                    )

        # Debug both datasets
        debug_conversation(train_data, "TRAINING")
        debug_conversation(val_data, "VALIDATION")

        print("\n" + "=" * 80)
        print("END DEBUG")
        print("=" * 80 + "\n")

    # ============================================================================
    # Apply filtering for validation data
    # ============================================================================
    def has_assistant_response(example):
        conv = example["conversations"]
        return len(conv) >= 2 and conv[-1]["role"] == "assistant"

    print("Filtering validation data...")
    print(f"Val data before filtering: {len(val_data)}")
    val_data = val_data.filter(has_assistant_response)
    print(f"Val data after filtering: {len(val_data)}")

    if len(val_data) == 0:
        print("\n❌ WARNING: No validation examples left after filtering!")
        print("   Your validation data doesn't have assistant responses.")
        print("   Consider fixing your data loading or skipping validation.\n")

    # Format datasets
    print("\nFormatting and tokenizing datasets...")

    # Format datasets
    print("\nFormatting and tokenizing datasets...")

    train_data = train_data.map(tokenize_func, batched=True)
    val_data = val_data.map(tokenize_func, batched=True)

    # Fix for "AttributeError: 'list' object has no attribute 'tolist'"
    # Unsloth's train_on_responses_only expects numpy arrays, not python lists.
    # We only format the tensor columns so 'claim_id' remains a string for the sampler.
    tensor_cols = ["input_ids", "attention_mask", "labels"]
    train_data.set_format(
        type="numpy", columns=tensor_cols, output_all_columns=True
    )
    val_data.set_format(
        type="numpy", columns=tensor_cols, output_all_columns=True
    )

    if debug:
        print("\n" + "=" * 80)
        print("DEBUGGING TOKENIZATION & MASKING")
        print("=" * 80)

        def debug_tokenization(dataset, dataset_name, num_samples=2):
            print(f"\n{dataset_name} Dataset:")

            for i in range(min(num_samples, len(dataset))):
                example = dataset[i]

                print(f"\n  --- Example {i} ---")
                print(f"  Input IDs length: {len(example['input_ids'])}")
                print(f"  Labels length: {len(example['labels'])}")
                print(
                    f"  Attention mask length: {len(example['attention_mask'])}"
                )

                # Count masked tokens
                labels = example["labels"]
                masked_count = sum(1 for x in labels if x == -100)
                trainable_count = sum(1 for x in labels if x != -100)

                print(f"  Masked tokens (instruction): {masked_count}")
                print(f"  Trainable tokens (response): {trainable_count}")
                print(
                    f"  Masking ratio: {masked_count / len(labels) * 100:.1f}%"
                )

                # Decode to show what's masked vs trainable
                input_ids = example["input_ids"]

                # Show first 10 tokens
                print(f"\n  First 10 tokens (should be masked):")
                for j in range(min(10, len(input_ids))):
                    token_id = input_ids[j]
                    label = labels[j]
                    token_text = tokenizer.decode([token_id])
                    mask_status = "MASKED" if label == -100 else "TRAIN"
                    print(
                        f"    [{j}] {mask_status:7s} | token_id={token_id:5d} | {repr(token_text)}"
                    )

                # Find transition from masked to trainable
                transition_idx = None
                for j in range(len(labels)):
                    if labels[j] != -100:
                        transition_idx = j
                        break

                if transition_idx:
                    print(
                        f"\n  Transition point (first trainable token at index {transition_idx}):"
                    )
                    start = max(0, transition_idx - 3)
                    end = min(len(input_ids), transition_idx + 3)
                    for j in range(start, end):
                        token_id = input_ids[j]
                        label = labels[j]
                        token_text = tokenizer.decode([token_id])
                        mask_status = "MASKED" if label == -100 else "TRAIN"
                        marker = (
                            " <-- TRANSITION" if j == transition_idx else ""
                        )
                        print(
                            f"    [{j}] {mask_status:7s} | token_id={token_id:5d} | {repr(token_text)}{marker}"
                        )

                # Decode full texts
                print(f"\n  Full input (decoded):")
                full_text = tokenizer.decode(input_ids)
                print(f"    {repr(full_text[:200])}...")

                # Decode only the trainable part
                trainable_ids = [
                    input_ids[j]
                    for j in range(len(input_ids))
                    if labels[j] != -100
                ]
                trainable_text = tokenizer.decode(trainable_ids)
                print(f"\n  Trainable portion (what model learns from):")
                print(f"    {repr(trainable_text)}")

        debug_tokenization(train_data, "TRAINING")
        if len(val_data) > 0:
            debug_tokenization(val_data, "VALIDATION")

        print("\n" + "=" * 80)
        print("END DEBUG")
        print("=" * 80 + "\n")

    # Create batch sampler BEFORE removing columns (needs claim_id, perturbation_type)
    pre_built_batch_sampler = None
    if batch_match_claim_id:
        print("\n" + "=" * 80)
        print(
            "🔧 Creating EnhancedPerturbationBatchSampler (before column removal)"
        )
        print("=" * 80)
        pre_built_batch_sampler = EnhancedPerturbationBatchSampler(
            dataset=train_data,
            batch_size=batch_size,
            batch_match_claim_id=batch_match_claim_id,
            split_large_claims=split_large_claims,
            ensure_original_in_batch=ensure_original_in_batch,
            ensure_perturbed_in_batch=ensure_perturbed_in_batch,
            shuffle=True,
            seed=3407 + start_epoch,
            drop_last=False,
        )
        print(
            f"✅ Batch sampler created with {len(pre_built_batch_sampler)} batches"
        )
        print("=" * 80 + "\n")

    # Remove ALL non-tensor columns - keep only 'text' for SFTTrainer
    # String columns like claim_id, perturbation_type, ground_truth cause tensor errors

    # If using the custom sampler, we also can to keep metadata
    # until after the sampler is initialized
    model_inputs = {"input_ids", "attention_mask", "labels", "text"}
    sampler_metadata = {"claim_id", "perturbation_type"}

    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    train_data = train_data.remove_columns(
        [c for c in train_data.column_names if c not in columns_to_keep]
    )
    val_data = val_data.remove_columns(
        [c for c in val_data.column_names if c not in columns_to_keep]
    )

    print(f"Final train columns: {train_data.column_names}")
    print(f"Final val columns: {val_data.column_names}")

    # Report truncation
    if truncated_count > 0:
        truncation_pct = 100 * truncated_count / total_count
        print(
            f"\n⚠️  WARNING: {truncated_count}/{total_count} examples ({truncation_pct:.1f}%) "
            f"exceed max_seq_length!"
        )

        if truncation_pct > 10:
            print(f"\n❌ ERROR: >10% of data will be truncated!")
            raise ValueError(
                "Too much data truncation - increase max_seq_length"
            )
    else:
        print(
            f"\n✅ All examples fit within max_seq_length ({max_seq_length:,} tokens)"
        )

    # Create output directory (used for checkpoints and final model)
    base_dir = f"{prefix}models/{model_name}_{model_name_timestamp}_{perturb_type_names}"
    output_dir = f"{base_dir}/checkpoints{rank_postfix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Calculate final epoch number
    target_epoch = start_epoch + num_train_epochs - 1

    # Initialize wandb
    run = wandb.init(
        project="numpert_finetune",
        name=f"{model_name}{quant_status}_epoch_{start_epoch}→{target_epoch}_(r{rank})_{perturb_type_names}",
        config={
            "model": model_name,
            "rank": rank,
            "start_epoch": start_epoch,
            "num_train_epochs": num_train_epochs,
            "target_epoch": target_epoch,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "expand_perturbations": expand_perturbations,
            "balance_labels": balance_labels,
            "perturb_ratio": perturb_ratio,
            "false_perturb_t_to_f_ratio": false_perturb_t_to_f_ratio,
            "batch_match_claim_id": batch_match_claim_id,
            "split_large_claims": split_large_claims,
            "ensure_original_in_batch": ensure_original_in_batch,
            "ensure_perturbed_in_batch": ensure_perturbed_in_batch,
            "truncated_examples": truncated_count,
            "truncation_rate": (
                truncated_count / total_count if total_count > 0 else 0
            ),
        },
    )

    training_config = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        num_train_epochs=num_train_epochs,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407 + start_epoch,
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # 3. Initialize the correct Trainer
    if batch_match_claim_id:
        trainer = PerturbationAwareSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_config,
            callbacks=[ValidationCallback()],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True
            ),
            use_custom_sampler=True,
            pre_built_batch_sampler=pre_built_batch_sampler,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_config,
            callbacks=[ValidationCallback()],
        )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    actual_final_epoch = start_epoch + int(trainer.state.epoch)

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    print("=" * 80 + "\n")

    run.finish()

    # Save model
    save_dir = f"{base_dir}/epoch-{actual_final_epoch}{quant_status}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    lora_path = f"{save_dir}/lora_weights{rank_postfix}"

    if not os.path.exists(lora_path):
        os.makedirs(lora_path, exist_ok=True)

    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA weights saved to: {lora_path}")

    # Merge and save
    merged_path = f"{save_dir}/merged_model{rank_postfix}"
    if not os.path.exists(merged_path):
        os.makedirs(merged_path, exist_ok=True)

    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {merged_path}")
    print(f"✅ Training complete")
    print(f"   Started at epoch: {start_epoch}")
    print(f"   Trained for: {int(trainer.state.epoch)} epochs")
    print(f"   Final epoch reached: {actual_final_epoch}")


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
    test_json_paths: List[str],
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
        # TODO: Må fikse den her til correct
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
    ground_truths = list(test_data["ground_truth"])
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
                    "ground_truth": test_data[idx]["ground_truth"],
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
                    "ground_truth": test_data[i]["ground_truth"],
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
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

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
        accuracy = accuracy_score(valid_ground_truths, valid_predictions)
        f1_macro = f1_score(
            valid_ground_truths,
            valid_predictions,
            average="macro",
            zero_division=0,
        )
        f1_weighted = f1_score(
            valid_ground_truths,
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
                valid_ground_truths, valid_predictions, labels=VALID_LABELS
            )
        )

    # Per-perturbation analysis (only on valid predictions)
    print("\nPer-Perturbation Type Analysis:")
    perturb_metrics = defaultdict(
        lambda: {"correct": 0, "total": 0, "failures": 0}
    )

    # Count failures per perturbation type
    for i in parse_failure_indices:
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["failures"] += 1

    # Count correct/total for valid predictions
    for i in valid_indices:
        pred = labels_extracted[i]
        gt = ground_truths[i]
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
            "ground_truth": ground_truths,
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
                "ground_truth": ground_truths[i],
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
        description="Fine-tune and evaluate fact-checking models"
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["finetune", "inference"],
        required=True,
        help="Mode: finetune or inference",
    )

    # Data paths
    parser.add_argument(
        "--train_json", nargs="+", help="Path(s) to training JSON files"
    )
    parser.add_argument(
        "--val_json", nargs="+", help="Path(s) to validation JSON files"
    )
    parser.add_argument(
        "--test_json", nargs="+", help="Path(s) to test JSON files"
    )

    # Model configuration
    parser.add_argument("--model", default=None, help="Model name")

    # Perturbation type name for model naming
    parser.add_argument(
        "--perturb_type_names",
        default=None,
        help="Adds perturb type to trained model name",
    )

    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Epoch number to resume at to resume",
    )
    parser.add_argument(
        "--resume_model",
        type=str,
        default=None,
        help="Path to a previously saved merged model to resume training from (required when start_epoch > 1)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
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
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Balance True/False label distribution",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=1.0,
        help="Ratio of perturbed to non-perturbed examples (1.0 = 1:1)",
    )
    parser.add_argument(
        "--false_perturb_t_to_f_ratio",
        type=float,
        default=0.67,
        help="Among False perturbed examples, ratio that should be T→F",
    )

    # Batch matching
    parser.add_argument(
        "--batch_match_claim_id",
        action="store_true",
        help="Try to group same claim_id in batches",
    )
    parser.add_argument(
        "--split_large_claims",
        action="store_true",
        default=True,
        help="Split claims with many perturbations across batches",
    )
    parser.add_argument(
        "--ensure_original_in_batch",
        action="store_true",
        default=True,
        help="Ensure each batch has at least one original example",
    )
    parser.add_argument(
        "--ensure_perturbed_in_batch",
        action="store_true",
        default=True,
        help="Ensure each batch has at least one perturbed example",
    )

    # Evaluation
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Checkpoint save frequency"
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for conversation format and tokenization/masking",
    )

    # Inference
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM for inference"
    )
    parser.add_argument(
        "--output_dir",
        default="predictions/numpert",
        help="Output directory for predictions",
    )

    args = parser.parse_args()

    if args.mode == "finetune":
        if not args.train_json or not args.val_json:
            parser.error(
                "--train_json and --val_json are required for finetuning"
            )

        finetune(
            train_json_paths=args.train_json,
            val_json_paths=args.val_json,
            model_name=args.model,
            perturb_type_names=args.perturb_type_names,
            rank=args.rank,
            start_epoch=args.start_epoch,
            resume_model=args.resume_model,
            num_train_epochs=args.num_train_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.max_seq_length,
            expand_perturbations=args.expand_perturbations,
            balance_labels=args.balance_labels,
            perturb_ratio=args.perturb_ratio,
            false_perturb_t_to_f_ratio=args.false_perturb_t_to_f_ratio,
            batch_match_claim_id=args.batch_match_claim_id,
            split_large_claims=args.split_large_claims,
            ensure_original_in_batch=args.ensure_original_in_batch,
            ensure_perturbed_in_batch=args.ensure_perturbed_in_batch,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            debug=args.debug,
        )

    elif args.mode == "inference":
        if not args.test_json:
            parser.error("--test_json is required for inference")

        inference(
            test_json_paths=args.test_json,
            model_name=args.model,
            rank=args.rank,
            start_epoch=args.start_epoch,
            use_vllm=args.use_vllm,
            expand_perturbations=args.expand_perturbations,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
        )
