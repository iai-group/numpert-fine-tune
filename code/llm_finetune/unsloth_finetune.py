import json
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict, Counter
import numpy as np
import random

from code.llm_finetune.hf_dataset_instructions import (
    load_fact_checking_data,
    load_train_and_validation,
    PerturbationType,
)
from code.llm_finetune.data_balancer import DatasetBalancer
from code.llm_finetune.batch_sampler import EnhancedPerturbationBatchSampler


from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer
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

# Valid labels for fact-checking
VALID_LABELS = ["True", "False"]


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
        batch_match_claim_id: bool = False,
        split_large_claims: bool = True,
        ensure_original_in_batch: bool = True,
        ensure_perturbed_in_batch: bool = True,
        sampler_seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            use_custom_sampler: Whether to use EnhancedPerturbationBatchSampler
            batch_match_claim_id: Group same claim_id in batches
            split_large_claims: Split large claims across batches
            ensure_original_in_batch: Ensure each batch has ≥1 original
            ensure_perturbed_in_batch: Ensure each batch has ≥1 perturbed
            sampler_seed: Random seed for sampler
        """
        super().__init__(*args, **kwargs)

        self.use_custom_sampler = use_custom_sampler
        self.batch_match_claim_id = batch_match_claim_id
        self.split_large_claims = split_large_claims
        self.ensure_original_in_batch = ensure_original_in_batch
        self.ensure_perturbed_in_batch = ensure_perturbed_in_batch
        self.sampler_seed = sampler_seed

    def get_train_dataloader(self) -> DataLoader:
        """Override to use custom batch sampler when enabled."""
        if not self.use_custom_sampler:
            # Use default SFTTrainer data loading
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        print("\n" + "=" * 80)
        print("🔧 Using EnhancedPerturbationBatchSampler for training")
        print("=" * 80)

        # Create custom batch sampler
        batch_sampler = EnhancedPerturbationBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            batch_match_claim_id=self.batch_match_claim_id,
            split_large_claims=self.split_large_claims,
            ensure_original_in_batch=self.ensure_original_in_batch,
            ensure_perturbed_in_batch=self.ensure_perturbed_in_batch,
            shuffle=True,
            seed=self.sampler_seed,
            drop_last=self.args.dataloader_drop_last,
        )

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
        return "unsloth/Qwen3-8B"
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

    # Load model
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
    train_label_dist = Counter(train_data["label"])
    print("\nTrain label distribution:")
    for label, count in train_label_dist.most_common():
        print(f"  {label:30s}: {count:5d} ({100*count/len(train_data):5.1f}%)")

    # Format datasets
    print("\nFormatting datasets...")
    train_data = train_data.map(formatting_prompts_func, batched=True)
    val_data = val_data.map(formatting_prompts_func, batched=True)

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
    output_dir = f"epoch-{epoch}/{model_name}{quant_status}/outputs/numpert{rank_postfix}"
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

    # Training arguments
    training_args = TrainingArguments(
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
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # Create trainer - use custom trainer if batch matching is enabled
    if batch_match_claim_id:
        trainer = PerturbationAwareSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=data_collator,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[ValidationCallback()],
            # Custom sampler parameters
            use_custom_sampler=True,
            batch_match_claim_id=batch_match_claim_id,
            split_large_claims=split_large_claims,
            ensure_original_in_batch=ensure_original_in_batch,
            ensure_perturbed_in_batch=ensure_perturbed_in_batch,
            sampler_seed=3407 + epoch,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=data_collator,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[ValidationCallback()],
        )

    # Train on responses only
    if "llama" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif "qwen" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="user\n",
            response_part="assistant\n",
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


def extract_label_from_response(text: str) -> str:
    """Extract the predicted label from model response."""
    text = text.strip()

    if "True" in text or "true" in text:
        return "True"
    if "False" in text or "false" in text:
        return "False"

    return random.choice(VALID_LABELS)


def inference(
    test_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    epoch: int = 0,
    use_vllm: bool = False,
    expand_perturbations: bool = True,
    max_seq_length: int = 16384,
    output_dir: str = "predictions/numpert",
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
    predictions = []
    ground_truths = test_data["ground_truth"]

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
            label = extract_label_from_response(decoded)
            predictions.append(label)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(
        ground_truths, predictions, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        ground_truths, predictions, average="weighted", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(ground_truths, predictions, labels=VALID_LABELS)
    )

    # Per-perturbation analysis
    print("\nPer-Perturbation Type Analysis:")
    perturb_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["total"] += 1
        if pred == gt:
            perturb_metrics[p_type]["correct"] += 1

    for p_type in sorted(perturb_metrics.keys()):
        metrics = perturb_metrics[p_type]
        acc = (
            metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        )
        print(
            f"  {p_type:30s}: {acc:.4f} ({metrics['correct']}/{metrics['total']})"
        )

    print("=" * 80 + "\n")

    # Save predictions
    results_df = pd.DataFrame(
        {
            "claim_id": test_data["claim_id"],
            "perturbation_type": test_data["perturbation_type"],
            "ground_truth": ground_truths,
            "prediction": predictions,
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
        "per_perturbation": {
            p_type: {
                "accuracy": m["correct"] / m["total"] if m["total"] > 0 else 0,
                "correct": m["correct"],
                "total": m["total"],
            }
            for p_type, m in perturb_metrics.items()
        },
    }

    metrics_path = f"{pred_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_path}\n")


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
