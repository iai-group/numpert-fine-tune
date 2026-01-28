import json
from typing import Optional, List, Dict
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
from code.llm_finetune.hf_dataset_instructions import load_train_and_validation
from code.llm_finetune.config import prefix
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import wandb
import argparse
import random
import os
import re
from tqdm import tqdm
import pandas as pd

# Valid labels for fact-checking
VALID_LABELS = ["True", "False"]


class PerturbationBatchSampler(Sampler):
    """
    Custom sampler that creates batches with controlled perturbation distribution.
    
    Features:
    - Can group examples with same claim_id in the same batch
    - Controls ratio of perturbed vs non-perturbed examples
    - Ensures diverse perturbation types in each batch
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        group_by_claim_id: bool = False,
        perturb_ratio: float = 0.5,  # Ratio of perturbed to non-perturbed
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Args:
            dataset: HuggingFace Dataset with 'claim_id' and 'perturbation_type' columns
            batch_size: Number of examples per batch
            group_by_claim_id: If True, tries to put same claim_id in same batch
            perturb_ratio: Target ratio of perturbed examples (0.0 to 1.0)
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_by_claim_id = group_by_claim_id
        self.perturb_ratio = perturb_ratio
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        
        # Group indices by claim_id and perturbation type
        self.claim_id_groups = defaultdict(list)
        self.original_indices = []
        self.perturbed_indices = []
        
        for idx in range(len(dataset)):
            claim_id = dataset[idx]['claim_id']
            perturb_type = dataset[idx]['perturbation_type']
            
            self.claim_id_groups[claim_id].append(idx)
            
            if perturb_type == 'original':
                self.original_indices.append(idx)
            else:
                self.perturbed_indices.append(idx)
        
        print(f"\n{'='*80}")
        print(f"PerturbationBatchSampler initialized:")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Unique claim_ids: {len(self.claim_id_groups)}")
        print(f"  Original examples: {len(self.original_indices)}")
        print(f"  Perturbed examples: {len(self.perturbed_indices)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Group by claim_id: {group_by_claim_id}")
        print(f"  Target perturb ratio: {perturb_ratio}")
        print(f"{'='*80}\n")
    
    def __iter__(self):
        if self.group_by_claim_id:
            batches = self._create_claim_grouped_batches()
        else:
            batches = self._create_perturbation_controlled_batches()
        
        if self.shuffle:
            self.rng.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def _create_claim_grouped_batches(self):
        """Create batches where examples with same claim_id are grouped together"""
        batches = []
        claim_ids = list(self.claim_id_groups.keys())
        
        if self.shuffle:
            self.rng.shuffle(claim_ids)
        
        current_batch = []
        for claim_id in claim_ids:
            indices = self.claim_id_groups[claim_id]
            
            # If adding this claim would exceed batch size, start new batch
            if len(current_batch) + len(indices) > self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = indices.copy()
            else:
                current_batch.extend(indices)
            
            # If batch is full, save it
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch[:self.batch_size])
                current_batch = current_batch[self.batch_size:]
        
        # Add remaining examples
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_perturbation_controlled_batches(self):
        """Create batches with controlled perturbation ratio"""
        batches = []
        
        original_pool = self.original_indices.copy()
        perturbed_pool = self.perturbed_indices.copy()
        
        if self.shuffle:
            self.rng.shuffle(original_pool)
            self.rng.shuffle(perturbed_pool)
        
        # Calculate how many of each type per batch
        n_perturbed = int(self.batch_size * self.perturb_ratio)
        n_original = self.batch_size - n_perturbed
        
        while len(original_pool) >= n_original and len(perturbed_pool) >= n_perturbed:
            batch = []
            batch.extend(original_pool[:n_original])
            batch.extend(perturbed_pool[:n_perturbed])
            
            original_pool = original_pool[n_original:]
            perturbed_pool = perturbed_pool[n_perturbed:]
            
            if self.shuffle:
                self.rng.shuffle(batch)
            
            batches.append(batch)
        
        # Handle remaining examples
        remaining = original_pool + perturbed_pool
        if remaining:
            if self.shuffle:
                self.rng.shuffle(remaining)
            for i in range(0, len(remaining), self.batch_size):
                batches.append(remaining[i:i + self.batch_size])
        
        return batches
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ValidationCallback(TrainerCallback):
    """Callback to run validation and print metrics during training"""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print("\n" + "="*80)
            print("VALIDATION METRICS:")
            print("="*80)
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("="*80 + "\n")


def get_model_name_arg(model_name):
    """Convert short model name to full HuggingFace path"""
    if model_name.startswith("Qwen") or model_name.startswith("qwen"):
        return "unsloth/Qwen3-8B"
    elif model_name.startswith("llama-3.2-3b"):
        return "unsloth/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b"):
        return "unsloth/Llama-3.2-1B-Instruct"
    return model_name


def compute_metrics_for_fact_checking(eval_pred):
    """
    Compute metrics for fact-checking evaluation.
    Used by Trainer for validation during training.
    """
    predictions, labels = eval_pred
    
    # If predictions are logits, get the argmax
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def finetune(
    train_json_paths: List[str],
    val_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    epoch: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    expand_perturbations: bool = True,
    group_by_claim_id: bool = False,
    perturb_ratio: float = 0.5,
    use_custom_sampler: bool = False,
    eval_steps: int = 100,
    save_steps: int = 500,
):
    """
    Fine-tune a model for fact-checking with numerical perturbations.
    
    Args:
        train_json_paths: List of paths to training JSON files
        val_json_paths: List of paths to validation JSON files
        model_name: Name of the model to fine-tune
        rank: LoRA rank
        epoch: Current epoch number
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        expand_perturbations: Whether to expand perturbations
        group_by_claim_id: Whether to group same claim_ids in batches
        perturb_ratio: Ratio of perturbed to non-perturbed in batches
        use_custom_sampler: Whether to use custom perturbation sampler
        eval_steps: How often to evaluate
        save_steps: How often to save checkpoints
    """
    
    print("\n" + "="*80)
    print("STARTING FINE-TUNING")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Rank: {rank}")
    print(f"Epoch: {epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Expand perturbations: {expand_perturbations}")
    print(f"Use custom sampler: {use_custom_sampler}")
    if use_custom_sampler:
        print(f"  Group by claim_id: {group_by_claim_id}")
        print(f"  Perturbation ratio: {perturb_ratio}")
    print("="*80 + "\n")
    
    # Model configuration
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    
    # Naming conventions
    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"
    
    # Determine model path
    if epoch == 1:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"{prefix}epoch-{epoch-1}/{model_name}{quant_status}/lora_numpert{rank_postfix}"
    
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
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
    
    # Formatting function
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                 for convo in convos]
        return {"text": texts}
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_train_and_validation(
        train_json_paths=train_json_paths,
        val_json_paths=val_json_paths,
        expand_perturbations_flag=expand_perturbations,
        include_doc=True
    )
    
    train_data = datasets["train"]
    val_data = datasets["validation"]
    
    print(f"Train dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")
    
    # Show perturbation distribution
    from collections import Counter
    train_perturb_dist = Counter(train_data['perturbation_type'])
    print("\nTrain perturbation distribution:")
    for p_type, count in train_perturb_dist.most_common():
        print(f"  {p_type:30s}: {count:5d} ({100*count/len(train_data):5.1f}%)")
    
    # Format datasets
    train_data = train_data.map(formatting_prompts_func, batched=True)
    val_data = val_data.map(formatting_prompts_func, batched=True)
    
    # Create output directory
    output_dir = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/numpert{rank_postfix}"
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
            "expand_perturbations": expand_perturbations,
            "use_custom_sampler": use_custom_sampler,
            "group_by_claim_id": group_by_claim_id,
            "perturb_ratio": perturb_ratio,
        }
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    # Create trainer
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
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    print("="*80 + "\n")
    
    run.finish()
    
    # Save model
    save_dir = f"{prefix}epoch-{epoch}/{model_name}{quant_status}"
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
    """
    Extract the predicted label from model response.
    
    Args:
        text: Generated text from model
        
    Returns:
        Extracted label ("True" or "False")
    """
    text = text.strip()
    
    # Direct match
    if "True" in text or "true" in text:
        return "True"
    if "False" in text or "false" in text:
        return "False"
    
    # Fallback to random choice
    return random.choice(VALID_LABELS)


def inference(
    test_json_paths: List[str],
    model_name: str = "llama-3.2-3b-instruct",
    rank: int = 32,
    epoch: int = 0,
    use_vllm: bool = False,
    expand_perturbations: bool = True,
    output_dir: str = "predictions/numpert"
):
    """
    Run inference on test data.
    
    Args:
        test_json_paths: List of paths to test JSON files
        model_name: Name of the model
        rank: LoRA rank
        epoch: Epoch to load (0 for base model)
        use_vllm: Whether to use vLLM for inference
        expand_perturbations: Whether to expand perturbations
        output_dir: Directory to save predictions
    """
    
    print("\n" + "="*80)
    print("STARTING INFERENCE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Epoch: {epoch}")
    print(f"Use vLLM: {use_vllm}")
    print(f"Expand perturbations: {expand_perturbations}")
    print("="*80 + "\n")
    
    # Configuration
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    
    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"
    
    # Determine model path
    if epoch == 0:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_numpert{rank_postfix}"
    
    print(f"Loading model from: {model_name_arg}\n")
    
    # Load model
    if use_vllm:
        llm = LLM(model=model_name_arg, max_model_len=max_seq_length)
        tokenizer = AutoTokenizer.from_pretrained(model_name_arg, trust_remote_code=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_arg,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            fix_tokenizer=False
        )
        FastLanguageModel.for_inference(model)
    
    # Load test data
    print("Loading test data...")
    test_data = load_fact_checking_data(
        json_paths=test_json_paths,
        split="test",
        expand_perturbations_flag=expand_perturbations,
        include_doc=True
    )
    
    print(f"Test dataset size: {len(test_data)}\n")
    
    # Prepare for predictions
    predictions = []
    ground_truths = test_data["ground_truth"]
    
    # Create output directory
    vllm_suffix = "-vllm" if use_vllm else ""
    pred_dir = f"{output_dir}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_suffix}"
    os.makedirs(pred_dir, exist_ok=True)
    
    # Run inference
    print("Running inference...")
    
    if use_vllm:
        # vLLM batch inference
        sampling_params = SamplingParams(temperature=0, max_tokens=512, seed=42)
        prompts = [tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        ) for conv in test_data["conversations"]]
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            text = output.outputs[0].text
            label = extract_label_from_response(text)
            predictions.append(label)
    
    else:
        # Regular inference
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
                    model_inputs["attention_mask"] = (model_inputs["input_ids"] != 0).long()
                else:
                    model_inputs["attention_mask"] = (model_inputs["input_ids"] != pad_id).long()
            
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=512,
                use_cache=False,
                temperature=0.001,
            )
            
            decoded = tokenizer.batch_decode(outputs)[0]
            label = extract_label_from_response(decoded)
            predictions.append(label)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(ground_truths, predictions, labels=VALID_LABELS))
    
    # Per-perturbation type analysis
    print("\nPer-Perturbation Type Analysis:")
    perturb_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        p_type = test_data[i]["perturbation_type"]
        perturb_metrics[p_type]["total"] += 1
        if pred == gt:
            perturb_metrics[p_type]["correct"] += 1
    
    for p_type in sorted(perturb_metrics.keys()):
        metrics = perturb_metrics[p_type]
        acc = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        print(f"  {p_type:30s}: {acc:.4f} ({metrics['correct']}/{metrics['total']})")
    
    print("="*80 + "\n")
    
    # Save predictions
    results_df = pd.DataFrame({
        "claim_id": test_data["claim_id"],
        "perturbation_type": test_data["perturbation_type"],
        "ground_truth": ground_truths,
        "prediction": predictions,
    })
    
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
                "total": m["total"]
            }
            for p_type, m in perturb_metrics.items()
        }
    }
    
    metrics_path = f"{pred_dir}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate fact-checking models")
    
    parser.add_argument("--mode", choices=["finetune", "inference"], required=True,
                       help="Mode: finetune or inference")
    parser.add_argument("--train_json", nargs="+", help="Path(s) to training JSON files")
    parser.add_argument("--val_json", nargs="+", help="Path(s) to validation JSON files")
    parser.add_argument("--test_json", nargs="+", help="Path(s) to test JSON files")
    parser.add_argument("--model", default="llama-3.2-3b-instruct",
                       help="Model name")
    parser.add_argument("--rank", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--epoch", type=int, default=1,
                       help="Epoch number")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--expand_perturbations", action="store_true", default=True,
                       help="Expand perturbations in dataset")
    parser.add_argument("--group_by_claim_id", action="store_true",
                       help="Group same claim_ids in batches")
    parser.add_argument("--perturb_ratio", type=float, default=0.5,
                       help="Ratio of perturbed examples in batches")
    parser.add_argument("--use_custom_sampler", action="store_true",
                       help="Use custom perturbation batch sampler")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for inference")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Checkpoint save frequency")
    
    args = parser.parse_args()
    
    if args.mode == "finetune":
        if not args.train_json or not args.val_json:
            parser.error("--train_json and --val_json are required for finetuning")
        
        finetune(
            train_json_paths=args.train_json,
            val_json_paths=args.val_json,
            model_name=args.model,
            rank=args.rank,
            epoch=args.epoch,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            expand_perturbations=args.expand_perturbations,
            group_by_claim_id=args.group_by_claim_id,
            perturb_ratio=args.perturb_ratio,
            use_custom_sampler=args.use_custom_sampler,
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
        )