#!/usr/bin/env python3
"""Minimal Unsloth training + GGUF export demo.

This script demonstrates:
1. Loading a tiny Unsloth model
2. Fine-tuning on sample data
3. Merging adapter to base model
4. Exporting to GGUF format

Run with:
    uv run python scripts/demo_unsloth_to_gguf.py
"""

import os
import sys
import torch
from pathlib import Path

from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from unsloth import FastLanguageModel

from src.llm_finetune.utils.training.convert_to_gguf import (
    convert_existing_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_sample_dataset():
    """Create tiny sample dataset for demo."""
    data = {
        "text": [
            "How do you add two numbers? You add by using the + operator.",
            "What is 2 + 2? 2 + 2 equals 4.",
            "Tell me a fact. The Earth revolves around the Sun.",
            "What is the capital of France? The capital of France is Paris.",
            (
                "How do you multiply numbers? "
                "Multiplication combines groups of numbers."
            ),
        ]
    }
    return Dataset.from_dict(data)


def main():
    """Execute minimal Unsloth training and GGUF export demo.

    Demonstrates:
    1. Loading a tiny Unsloth model
    2. Fine-tuning on sample data
    3. Merging adapter to base model
    4. Exporting to GGUF format
    """
    print("\n" + "=" * 80)
    print("UNSLOTH MINIMAL DEMO -> GGUF CONVERSION")
    print("=" * 80)
    print(f"\n📍 Current working directory: {Path.cwd()}")

    # Output directories
    output_dir = Path("./models/demo_unsloth")
    merged_dir = output_dir / "merged"
    gguf_dir = output_dir / "gguf"

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # ============================================================================
    # 1. Load tiny Unsloth model
    # ============================================================================
    print("\n[1/4] Loading Unsloth model (TinyLlama 1.1B)...")

    model_name = "unsloth/tinyllama-bnb-4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # Setup for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("✅ Model loaded: {}".format(model_name))
    print("   Config: 4-bit, LoRA r=16, max_seq=512")

    # ============================================================================
    # 2. Prepare dataset and trainer
    # ============================================================================
    print("\n[2/4] Creating sample dataset and trainer...")

    dataset = create_sample_dataset()
    print(f"✅ Sample dataset created: {len(dataset)} examples")

    # Apply chat template
    # tokenizer = get_chat_template(
    #     tokenizer, chat_template="unsloth"
    # )  # or another supported template

    # Minimal training config
    training_args = SFTConfig(
        per_device_train_batch_size=2,
        max_steps=10,  # Very short for demo
        learning_rate=2e-4,
        logging_steps=5,
        output_dir=str(output_dir / "checkpoint"),
        save_steps=1000,  # Don't save intermediate checkpoints
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=512,
        packing=False,
    )

    print("✅ Trainer ready")

    # ============================================================================
    # 3. Train and merge
    # ============================================================================
    print("\n[3/4] Fine-tuning model (10 steps)...")

    trainer.train()
    print("✅ Training complete")

    print("\n   Merging adapter with base model...")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )
    print(f"✅ Merged model saved to: {merged_dir}")

    # ============================================================================
    # 4. Convert to GGUF
    # ============================================================================
    print("\n[4/4] Converting to GGUF format...")

    try:
        gguf_path = convert_existing_model(
            model_path=str(merged_dir),
            output_dir=str(gguf_dir),
            quantization_method="q4_k_m",
            max_seq_length=512,
        )
        print("✅ GGUF conversion succeeded!")
        print("   Output: {}".format(gguf_path))

    except RuntimeError as e:
        print("\n⚠️  GGUF conversion failed (expected if llama.cpp not setup):")
        print("   {}".format(e))
        print("\n   To enable GGUF export, run:")
        print("   bash scripts/setup_llamacpp.sh")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n📊 Model files:")
    print("   Merged (HF):  {}".format(merged_dir))
    print("   GGUF (Quant): {}".format(gguf_dir))
    print()


if __name__ == "__main__":
    main()
