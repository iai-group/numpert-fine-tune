"""Fine-tuning and inference for fact-checking models with numerical perturbations."""
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
from vllm import LLM, SamplingParams
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


def get_model_name_arg(model_name):
    """Convert short model name to full HuggingFace path."""
    if model_name.startswith("Qwen") or model_name.startswith("qwen"):
        return "unsloth/Qwen3-4B"
    elif model_name.startswith("llama-3.2-3b"):
        return "unsloth/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b"):
        return "unsloth/Llama-3.2-1B-Instruct"
    return model_name


def finetune(
    train_json_paths: List[str],
    val_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    epoch: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
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
):

    """Fine-tune a model for fact-checking with numerical perturbations."""

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Rank: {rank}")
    print(f"Epoch: {epoch}")
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
    if epoch == 1:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"epoch-{epoch-1}/{model_name}{quant_status}/lora_numpert{rank_postfix}"

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
    if "llama" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    elif "qwen" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # Set padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Formatting function with length monitoring
    truncated_count = 0
    total_count = 0

    def formatting_prompts_func(examples):
        nonlocal truncated_count, total_count

        convos = examples["conversations"]
        texts = []

        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )

            # Check length
            tokens = tokenizer(
                text, return_tensors=None, add_special_tokens=True
            )
            token_count = len(tokens["input_ids"])
            total_count += 1

            if token_count > max_seq_length:
                truncated_count += 1
                if truncated_count <= 5:
                    print(
                        f"⚠️  Example {total_count} exceeds max_seq_length: "
                        f"{token_count:,} > {max_seq_length:,} tokens"
                    )

            texts.append(text)

        return {"text": texts}

    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False, # Collator handles dynamic padding
        )

    # Load datasets
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
            seed=3407 + epoch,
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
    
    # Format datasets
    print("\nFormatting and tokenizing datasets...")
    train_data = train_data.map(formatting_prompts_func, batched=True)
    val_data = val_data.map(formatting_prompts_func, batched=True)
    
    train_data = train_data.map(tokenize_func, batched=True)
    val_data = val_data.map(tokenize_func, batched=True)

    # Create batch sampler BEFORE removing columns (needs claim_id, perturbation_type)
    pre_built_batch_sampler = None
    if batch_match_claim_id:
        print("\n" + "=" * 80)
        print("🔧 Creating EnhancedPerturbationBatchSampler (before column removal)")
        print("=" * 80)
        pre_built_batch_sampler = EnhancedPerturbationBatchSampler(
            dataset=train_data,
            batch_size=batch_size,
            batch_match_claim_id=batch_match_claim_id,
            split_large_claims=split_large_claims,
            ensure_original_in_batch=ensure_original_in_batch,
            ensure_perturbed_in_batch=ensure_perturbed_in_batch,
            shuffle=True,
            seed=3407 + epoch,
            drop_last=False,
        )
        print(f"✅ Batch sampler created with {len(pre_built_batch_sampler)} batches")
        print("=" * 80 + "\n")


    
    # Remove ALL non-tensor columns - keep only 'text' for SFTTrainer
    # String columns like claim_id, perturbation_type, ground_truth cause tensor errors
    columns_to_keep = {"input_ids", "attention_mask", "labels", "text"}

    # If using the custom sampler, we also can to keep metadata 
    # until after the sampler is initialized
    model_inputs = {"input_ids", "attention_mask", "labels", "text"}
    sampler_metadata = {"claim_id", "perturbation_type"}

    cols_to_remove = [c for c in train_data.column_names if c not in columns_to_keep]

    train_data = train_data.remove_columns(cols_to_remove)
    val_data = val_data.remove_columns(cols_to_remove)
    
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

    # Create output directory
    output_dir = f"{prefix}tuned_models/epoch-{epoch}/{model_name}{quant_status}_{rank_postfix}_{model_name_timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    run = wandb.init(
        project="numpert_finetune",
        name=f"{model_name}{quant_status} numpert epoch-{epoch} (r{rank})",
        config={
            "model": model_name,
            "rank": rank,
            "epoch": epoch,
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
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407 + epoch,
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

    # 4. Apply Response-Only Loss (The Unsloth way)
    # This effectively implements completion_only_loss=True by masking the instruction
    if "llama" in model_name.lower():
        # Llama 3.1/3.2 Template specific strings
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif "qwen" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # Train
    print("\nStarting training...\n")
    trainer.train()

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
    save_dir = f"epoch-{epoch}/{model_name}{quant_status}"
    os.makedirs(save_dir, exist_ok=True)

    lora_path = f"{save_dir}/lora_numpert{rank_postfix}"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA weights saved to: {lora_path}")

    # Merge and save
    merged_path = f"{save_dir}/merged_numpert{rank_postfix}"
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {merged_path}")

    print("\nFine-tuning complete!\n")


def extract_label_from_response(text: str) -> Optional[str]:
    """Extract the predicted label from model response.
    
    Expects JSON format: {"label": true} or {"label": false}
    
    Returns:
        The extracted label in JSON format, or None if unparseable
    """
    import re
    
    text = text.strip()
    
    # Strip Qwen3 thinking tokens if present
    if "<think>" in text and "</think>" in text:
        # Remove everything between <think> and </think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Also handle case where thinking isn't closed
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    
    text_lower = text.lower()
    
    # Check for exact or near-exact JSON matches first
    # Handle variations like {"label": true}, {"label":true}, { "label": true }, etc.
    if re.search(r'\{\s*"label"\s*:\s*true\s*\}', text_lower):
        return '{"label": true}'
    if re.search(r'\{\s*"label"\s*:\s*false\s*\}', text_lower):
        return '{"label": false}'
    
    # Fallback: check for "label": true/false pattern without full JSON structure
    if re.search(r'"label"\s*:\s*true', text_lower):
        return '{"label": true}'
    if re.search(r'"label"\s*:\s*false', text_lower):
        return '{"label": false}'
    
    # Last resort: check for plain true/false (model might omit JSON structure)
    # Check false first to avoid substring issues
    if "false" in text_lower and "true" not in text_lower:
        return '{"label": false}'
    if "true" in text_lower and "false" not in text_lower:
        return '{"label": true}'

    # Could not parse
    return None


def inference(
    test_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    epoch: int = 0,
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
    print(f"Epoch: {epoch}")
    print(f"Max sequence length: {max_seq_length:,} tokens")
    print(f"Use vLLM: {use_vllm}")
    print(f"Expand perturbations: {expand_perturbations}")
    print("=" * 80 + "\n")

    # Configuration
    dtype = None
    load_in_4bit = True

    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"

    # Determine model path
    if epoch == 0:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"epoch-{epoch}/{model_name}{quant_status}/lora_numpert{rank_postfix}"

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
    pred_dir = f"{output_dir}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_suffix}"
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

        for output in outputs:
            text = output.outputs[0].text
            raw_outputs.append(text)
            label = extract_label_from_response(text)
            predictions.append(label)

    else:
        for i in tqdm(range(len(test_data))):
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

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.001,
            )

            decoded = tokenizer.batch_decode(outputs)[0]
            raw_outputs.append(decoded)
            label = extract_label_from_response(decoded)
            predictions.append(label)

    # Separate valid predictions from parse failures
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    parse_failure_indices = [i for i, pred in enumerate(predictions) if pred is None]
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    
    n_total = len(predictions)
    n_valid = len(valid_predictions)
    n_failures = len(parse_failure_indices)

    # Report parse failures
    print(f"\n📊 Parse Results:")
    print(f"   Successfully parsed: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"   Parse failures: {n_failures}/{n_total} ({100*n_failures/n_total:.1f}%)")
    
    if n_failures > 0:
        print(f"\n⚠️  WARNING: {n_failures} responses could not be parsed and are EXCLUDED from metrics.")
        print("    Check raw_outputs.json and predictions.csv for details.")

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
            valid_ground_truths, valid_predictions, average="macro", zero_division=0
        )
        f1_weighted = f1_score(
            valid_ground_truths, valid_predictions, average="weighted", zero_division=0
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(valid_ground_truths, valid_predictions, labels=VALID_LABELS)
        )

    # Per-perturbation analysis (only on valid predictions)
    print("\nPer-Perturbation Type Analysis:")
    perturb_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "failures": 0})

    # Count failures per perturbation type
    for i in parse_failure_indices:
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["failures"] += 1

    # Count correct/total for valid predictions
    for i in valid_indices:
        pred = predictions[i]
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
        failure_note = f" [+{metrics['failures']} unparsed]" if metrics["failures"] > 0 else ""
        print(
            f"  {p_type:30s}: {acc:.4f} ({metrics['correct']}/{metrics['total']}){failure_note}"
        )

    print("=" * 80 + "\n")

    # Save predictions (include None for failures, marked clearly)
    results_df = pd.DataFrame(
        {
            "claim_id": test_data["claim_id"],
            "perturbation_type": test_data["perturbation_type"],
            "ground_truth": ground_truths,
            "prediction": [p if p is not None else "PARSE_FAILURE" for p in predictions],
            "parsed_successfully": [p is not None for p in predictions],
            "raw_output": raw_outputs,
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
        "note": f"Metrics computed on {n_valid}/{n_total} parseable responses only",
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
    parser.add_argument(
        "--model", default="llama-3.2-3b-instruct", help="Model name"
    )
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number")

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
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
            rank=args.rank,
            epoch=args.epoch,
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
        )

    elif args.mode == "inference":
        if not args.test_json:
            parser.error("--test_json is required for inference")

        inference(
            test_json_paths=args.test_json,
            model_name=args.model,
            rank=args.rank,
            epoch=args.epoch,
            use_vllm=args.use_vllm,
            expand_perturbations=args.expand_perturbations,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
        )