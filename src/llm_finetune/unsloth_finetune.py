"""Fine-tuning for fact-checking models with numerical perturbations."""

from typing import Optional, List
from torch.utils.data import DataLoader
from torch import torch
from collections import Counter
from datetime import datetime

from src.llm_finetune.config import prefix
from src.llm_finetune.utils.data_loading.hf_dataset_instructions import (
    load_train_and_validation,
    FileLoadConfig,
    parse_file_config,
)
from src.llm_finetune.utils.data_loading.class_balancer import DatasetBalancer
from src.llm_finetune.batch_sampler import EnhancedPerturbationBatchSampler
from src.llm_finetune.utils.training.convert_to_gguf import (
    save_as_gguf,
    DEFAULT_QUANT_METHOD,
    SUPPORTED_QUANT_METHODS,
)
from src.llm_finetune.utils.training.training_config_export import (
    build_training_config,
    save_training_config,
)


from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import (
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from unsloth import is_bfloat16_supported
import wandb
import argparse
import os

# Timestamp for model naming
time_start = datetime.now()
model_name_timestamp = time_start.strftime("%Y%m%d_%H%M%S")


class PerturbationAwareSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that uses EnhancedPerturbationBatchSampler.

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
        """Initialize the trainer with optional custom batch sampler.

        Args:
            use_custom_sampler: Whether to use EnhancedPerturbationBatchSampler
            pre_built_batch_sampler: Pre-built batch sampler (created
                before columns removed)
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
        """Print validation metrics at evaluation time."""
        if metrics:
            print("\n" + "=" * 80)
            print("VALIDATION METRICS:")
            print("=" * 80)
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("=" * 80 + "\n")


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


def finetune(  # noqa: C901
    train_json_paths: Optional[List[str]] = None,
    val_json_paths: Optional[List[str]] = None,
    train_file_configs: Optional[List[FileLoadConfig]] = None,
    val_file_configs: Optional[List[FileLoadConfig]] = None,
    model_name: str = "unsloth/Qwen3-8B",
    perturb_type_names: str = "",
    rank: int = 32,
    num_train_epochs: int = 1,
    start_epoch: int = 0,
    resume_model: Optional[str] = None,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 16384,
    expand_perturbations: bool = True,
    balance_labels: bool = False,
    perturb_ratio: float = 1.0,
    false_perturb_t_to_f_ratio: float = 0.67,
    equal_perturbation_methods: bool = False,
    batch_match_claim_id: bool = False,
    split_large_claims: bool = True,
    ensure_original_in_batch: bool = True,
    ensure_perturbed_in_batch: bool = True,
    eval_steps: int = 500,
    save_steps: int = 500,
    save_gguf: bool = False,
    gguf_quantization_method: str = DEFAULT_QUANT_METHOD,
    debug: bool = False,
):
    """Fine-tune a model for fact-checking with numerical perturbations."""
    target_epoch = start_epoch + num_train_epochs - 1

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
    print(f"Equal perturbation methods: {equal_perturbation_methods}")
    print(f"Batch match claim_id: {batch_match_claim_id}")
    if batch_match_claim_id:
        print(f"  - Split large claims: {split_large_claims}")
        print(f"  - Ensure original in batch: {ensure_original_in_batch}")
        print(f"  - Ensure perturbed in batch: {ensure_perturbed_in_batch}")
    print("=" * 80 + "\n")

    # Model configuration
    dtype = torch.bfloat16 if is_bfloat16_supported() else None
    load_in_4bit = True

    # Naming conventions
    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"

    # Determine model path
    if start_epoch == 0:
        model_name_arg = get_model_name_arg(model_name)
    else:
        if resume_model is None:
            raise ValueError(
                "--resume_model is required when start_epoch > 0. "
                "Provide the path to a previously saved merged model."
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
    else:  # TODO add more models and their templates as needed
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
                add_generation_prompt=False,  # EOS token: use False
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

            # Only unmask tokens from the end of the prompt to the
            # end of the full sequence
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
        train_file_configs=train_file_configs,
        val_file_configs=val_file_configs,
        expand_perturbations_flag=expand_perturbations,
        include_doc=True,
    )

    train_data = datasets["train"]
    val_data = datasets["validation"]

    print(f"Loaded train dataset size: {len(train_data)}")
    print(f"Loaded validation dataset size: {len(val_data)}")

    # Apply balancing to training data
    if balance_labels or perturb_ratio != 1.0 or equal_perturbation_methods:
        balancer = DatasetBalancer(
            dataset=train_data,
            balance_labels=balance_labels,
            perturb_ratio=perturb_ratio,
            false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
            equal_perturbation_methods=equal_perturbation_methods,
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
    train_label_dist = Counter(train_data["label"])
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
                "  Conversations WITHOUT assistant"
                f" response: {missing_assistant}"
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
                print(f"  Label: {example.get('label', 'N/A')}")
                print("  Conversation structure:")

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
                    print("  ✅ Last message is from assistant")
                else:
                    print(
                        f"  ❌ Last message is NOT from"
                        f" assistant (role: {conv[-1]['role']})"
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
    # We only format the tensor columns so 'claim_id' remains a
    # string for the sampler.
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
                print("\n  First 10 tokens (should be masked):")
                for j in range(min(10, len(input_ids))):
                    token_id = input_ids[j]
                    label = labels[j]
                    token_text = tokenizer.decode([token_id])
                    mask_status = "MASKED" if label == -100 else "TRAIN"
                    print(
                        f"    [{j}] {mask_status:7s} | "
                        f"token_id={token_id:5d} | "
                        f"{repr(token_text)}"
                    )

                # Find transition from masked to trainable
                transition_idx = None
                for j in range(len(labels)):
                    if labels[j] != -100:
                        transition_idx = j
                        break

                if transition_idx:
                    print(
                        f"\n  Transition point (first trainable"
                        f" token at index {transition_idx}):"
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
                            f"    [{j}] {mask_status:7s} | "
                            f"token_id={token_id:5d} | "
                            f"{repr(token_text)}{marker}"
                        )

                # Decode full texts
                print("\n  Full input (decoded):")
                full_text = tokenizer.decode(input_ids)
                print(f"    {repr(full_text[:200])}...")

                # Decode only the trainable part
                trainable_ids = [
                    input_ids[j]
                    for j in range(len(input_ids))
                    if labels[j] != -100
                ]
                trainable_text = tokenizer.decode(trainable_ids)
                print("\n  Trainable portion (what model learns from):")
                print(f"    {repr(trainable_text)}")

        debug_tokenization(train_data, "TRAINING")
        if len(val_data) > 0:
            debug_tokenization(val_data, "VALIDATION")

        print("\n" + "=" * 80)
        print("END DEBUG")
        print("=" * 80 + "\n")

    # Create batch sampler BEFORE removing columns
    # (needs claim_id, perturbation_type)
    pre_built_batch_sampler = None
    if batch_match_claim_id:
        print("\n" + "=" * 80)
        print(
            "\U0001f527 Creating EnhancedPerturbationBatchSampler"
            " (before column removal)"
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
            f"\u2705 Batch sampler created with"
            f" {len(pre_built_batch_sampler)} batches"
        )
        print("=" * 80 + "\n")

    # Remove ALL non-tensor columns - keep only 'text' for SFTTrainer
    # String columns like claim_id, perturbation_type, label cause tensor errors

    # If using the custom sampler, we also can to keep metadata
    # until after the sampler is initialized
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
            f"\n\u26a0\ufe0f  WARNING: {truncated_count}/{total_count} examples"
            f" ({truncation_pct:.1f}%) exceed max_seq_length!"
        )

        if truncation_pct > 10:
            print("\n❌ ERROR: >10% of data will be truncated!")
            raise ValueError(
                "Too much data truncation - increase max_seq_length"
            )
    else:
        print(
            "\n\u2705 All examples fit within max_seq_length"
            f" ({max_seq_length:,} tokens)"
        )

    # Create output directory (used for checkpoints and final model)
    base_dir = (
        f"{prefix}models/"
        f"{model_name}_{model_name_timestamp}_{perturb_type_names}"
    )
    output_dir = f"{base_dir}/checkpoints{rank_postfix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Export training configuration to JSON
    training_config_dict = build_training_config(
        model_name=model_name,
        train_file_configs=train_file_configs,
        val_file_configs=val_file_configs,
        train_dataset=train_data,
        val_dataset=val_data,
        rank=rank,
        num_train_epochs=num_train_epochs,
        start_epoch=start_epoch,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        expand_perturbations=expand_perturbations,
        balance_labels=balance_labels,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        equal_perturbation_methods=equal_perturbation_methods,
        batch_match_claim_id=batch_match_claim_id,
        split_large_claims=split_large_claims,
        ensure_original_in_batch=ensure_original_in_batch,
        ensure_perturbed_in_batch=ensure_perturbed_in_batch,
        save_gguf=save_gguf,
        gguf_quantization_method=gguf_quantization_method,
        timestamp=time_start,
    )
    config_path = save_training_config(training_config_dict, base_dir)
    print(f"Training config saved to: {config_path}")

    # Calculate final epoch number
    target_epoch = start_epoch + num_train_epochs - 1

    # Initialize wandb
    run = wandb.init(
        project="numpert_finetune",
        name=(
            f"{model_name}{quant_status}_epoch_"
            f"{start_epoch}→{target_epoch}"
            f"_(r{rank})_{perturb_type_names}"
        ),
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
            "equal_perturbation_methods": equal_perturbation_methods,
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

    # Optionally save GGUF quantized version
    if save_gguf:
        gguf_path = f"{save_dir}/gguf_{gguf_quantization_method}{rank_postfix}"
        save_as_gguf(
            model,
            tokenizer,
            output_dir=gguf_path,
            quantization_method=gguf_quantization_method,
        )

    print("✅ Training complete")
    print(f"   Started at epoch: {start_epoch}")
    print(f"   Trained for: {int(trainer.state.epoch)} epochs")
    print(f"   Final epoch reached: {actual_final_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune fact-checking models with numerical perturbations"
        ),
    )

    # Data paths — supports per-file perturbation type control.
    # Format: path/to/file.json[:type1,type2,...] (see parse_file_config)
    parser.add_argument(
        "--train_json",
        nargs="+",
        required=True,
        help=(
            "Path(s) to training JSON files. Supports per-file type control: "
            "path.json:type1,type2  (see FileLoadConfig)"
        ),
    )
    parser.add_argument(
        "--val_json",
        nargs="+",
        required=True,
        help=("Path(s) to validation JSON files. Same format as --train_json."),
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
        help=(
            "Path to a previously saved merged model to resume training"
            " from (required when start_epoch > 1)"
        ),
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
    parser.add_argument(
        "--equal_perturbation_methods",
        action="store_true",
        help=(
            "Stratify balancing so each perturbation method "
            "(numeration, range, …) gets an equal share of the budget"
        ),
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

    # GGUF quantization
    parser.add_argument(
        "--save_gguf",
        action="store_true",
        help="Save a GGUF-quantized copy alongside the merged model.",
    )
    parser.add_argument(
        "--gguf_quantization_method",
        type=str,
        default=DEFAULT_QUANT_METHOD,
        choices=SUPPORTED_QUANT_METHODS,
        help=f"GGUF quantization method (default: {DEFAULT_QUANT_METHOD}).",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable debug output for conversation format"
            " and tokenization/masking"
        ),
    )

    args = parser.parse_args()

    # Parse file configs (supports path:type1,type2 notation)
    train_configs = [parse_file_config(p) for p in args.train_json]
    val_configs = [parse_file_config(p) for p in args.val_json]

    finetune(
        train_file_configs=train_configs,
        val_file_configs=val_configs,
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
        equal_perturbation_methods=args.equal_perturbation_methods,
        batch_match_claim_id=args.batch_match_claim_id,
        split_large_claims=args.split_large_claims,
        ensure_original_in_batch=args.ensure_original_in_batch,
        ensure_perturbed_in_batch=args.ensure_perturbed_in_batch,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_gguf=args.save_gguf,
        gguf_quantization_method=args.gguf_quantization_method,
        debug=args.debug,
    )
