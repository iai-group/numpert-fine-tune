"""Sanity check: verify inference prompts do NOT contain the gold label.

Loads test data through the same pipeline as training and inference, applies
the chat template with ``conv[:-1]`` (the inference pattern), and asserts
the gold label does not appear in the formatted prompt.

Also shows side-by-side what the **training** full text looks like vs the
**inference** prompt so you can visually confirm the difference.

Usage::

    python -m tests.sanity_checks.check_chat_template_no_label_leak \
        --test_json data/inference_mini_test.json

    # With a specific model tokenizer (default: unsloth/Qwen3-8B)
    python -m tests.sanity_checks.check_chat_template_no_label_leak \
        --test_json data/inference_mini_test.json \
        --model_name unsloth/Qwen3-8B

    # Check a perturbed split file
    python -m tests.sanity_checks.check_chat_template_no_label_leak \
        --test_json data/perturbed_data/test_split/approximation_True.json
"""

from __future__ import annotations

import argparse
import sys

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from src.llm_finetune.utils.data_loading.pipeline import load_split

SEP = "=" * 80
THIN_SEP = "-" * 60


def check_prompts(
    test_json: str,
    model_name: str = "unsloth/Qwen3-8B",
    max_seq_length: int = 8192,
    show: int = 3,
) -> bool:
    """Load data and verify no gold label leaks into inference prompts.

    Returns True if all checks pass, False otherwise.
    """
    # Load tokenizer (no GPU needed, just the tokenizer)
    print(f"Loading tokenizer from: {model_name}")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        fix_tokenizer=False,
    )

    # Apply chat template (match training code)
    if "llama-3" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    elif "qwen" in model_name.lower():
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    else:
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # Load test data through the standard pipeline
    print(f"Loading test data from: {test_json}")
    test_data = load_split(
        json_paths=[test_json],
        split="test",
        use_perturbations=True,
        include_doc=True,
    )
    print(f"Loaded {len(test_data)} examples\n")

    leaked = 0
    checked = 0

    for i, example in enumerate(test_data):
        conv = example["conversations"]
        gold_label = example["label"]  # e.g. '{"label": true}'

        # ---- Inference prompt (conv[:-1]) ----
        inference_prompt = tokenizer.apply_chat_template(
            conv[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        # ---- Training full text (all turns) ----
        training_full = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Check: gold label must NOT appear in inference prompt
        has_leak = gold_label in inference_prompt
        if has_leak:
            leaked += 1
        checked += 1

        # Show first N examples regardless, and any that leak
        if i < show or has_leak:
            status = "LEAK" if has_leak else "OK"
            print(SEP)
            print(
                f"  Example {i} [{status}]  "
                f"claim_id={example['claim_id']}  "
                f"type={example['perturbation_type']}  "
                f"gold={gold_label}"
            )
            print(SEP)

            print(f"\n  [INFERENCE PROMPT] (conv[:-1], add_generation_prompt=True)")
            print(THIN_SEP)
            # Show last 400 chars to focus on the boundary
            tail = inference_prompt[-400:]
            if len(inference_prompt) > 400:
                tail = "..." + tail
            print(tail)

            print(f"\n  [TRAINING FULL TEXT] (all turns, add_generation_prompt=False)")
            print(THIN_SEP)
            tail = training_full[-400:]
            if len(training_full) > 400:
                tail = "..." + tail
            print(tail)
            print()

    # Summary
    print(SEP)
    print(f"  RESULTS: {checked} examples checked")
    print(SEP)

    if leaked > 0:
        print(f"\n  FAIL: {leaked}/{checked} examples have gold label in the inference prompt!")
        print("  The model sees the answer before generating.\n")
        return False
    else:
        print(f"\n  PASS: No gold label leakage detected in any inference prompt.")
        print(f"  All {checked} examples correctly strip the assistant turn.\n")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that inference prompts don't contain the gold label."
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to a test JSON file (e.g. data/inference_mini_test.json)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-8B",
        help="HuggingFace model name for the tokenizer (default: unsloth/Qwen3-8B)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Max sequence length (default: 8192)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="Number of examples to display (default: 3)",
    )
    args = parser.parse_args()

    ok = check_prompts(
        test_json=args.test_json,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        show=args.show,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
