"""Utility functions for analyzing sequence lengths in datasets."""
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import numpy as np

from src.llm_finetune.utils.data_loading.pipeline import load_split


def analyze_sequence_lengths(
    json_paths: List[str],
    tokenizer,
    split: str = "train",
    expand_perturbations: bool = True,
    max_examples: Optional[int] = None,
) -> Dict:
    """Analyze the distribution of sequence lengths in your dataset.

    Returns:
        Dictionary with statistics and length distribution
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING SEQUENCE LENGTHS - {split.upper()}")
    print("=" * 80)

    dataset = load_split(
        json_paths=json_paths,
        split=split,
        expand_perturbations=expand_perturbations,
        include_doc=True,
    )

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(f"Analyzing {len(dataset)} examples...")

    # Measure lengths
    sequence_lengths = []
    claim_lengths = []
    doc_lengths = []
    too_long_examples = []

    for i, example in enumerate(dataset):
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize to get actual length
        tokens = tokenizer(
            formatted, return_tensors=None, add_special_tokens=True
        )
        seq_len = len(tokens["input_ids"])
        sequence_lengths.append(seq_len)

        # Also measure claim and doc separately
        claim = example["conversations"][2]["content"]  # User's claim message
        claim_tokens = len(tokenizer(claim, return_tensors=None)["input_ids"])
        claim_lengths.append(claim_tokens)

        # Extract doc length if present
        if "Evidence:" in claim:
            doc_text = claim.split("Evidence:")[1].strip()
            doc_tokens = len(
                tokenizer(doc_text, return_tensors=None)["input_ids"]
            )
            doc_lengths.append(doc_tokens)

        # Track very long examples
        if seq_len > 8192:
            too_long_examples.append(
                {
                    "index": i,
                    "claim_id": example["claim_id"],
                    "length": seq_len,
                    "perturbation": example["perturbation_type"],
                }
            )

    # Calculate statistics
    stats = {
        "total_examples": len(dataset),
        "min_length": min(sequence_lengths),
        "max_length": max(sequence_lengths),
        "mean_length": np.mean(sequence_lengths),
        "median_length": np.median(sequence_lengths),
        "p95_length": np.percentile(sequence_lengths, 95),
        "p99_length": np.percentile(sequence_lengths, 99),
        "mean_claim_length": np.mean(claim_lengths) if claim_lengths else 0,
        "mean_doc_length": np.mean(doc_lengths) if doc_lengths else 0,
    }

    # Print statistics
    print("\nSequence Length Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Min length: {stats['min_length']:,} tokens")
    print(f"  Max length: {stats['max_length']:,} tokens")
    print(f"  Mean length: {stats['mean_length']:,.1f} tokens")
    print(f"  Median length: {stats['median_length']:,.1f} tokens")
    print(f"  95th percentile: {stats['p95_length']:,.1f} tokens")
    print(f"  99th percentile: {stats['p99_length']:,.1f} tokens")

    if claim_lengths:
        print(
            f"\n  Mean claim length: {stats['mean_claim_length']:,.1f} tokens"
        )
    if doc_lengths:
        print(
            "  Mean evidence doc length:"
            f" {stats['mean_doc_length']:,.1f} tokens"
        )

    # Show distribution
    print("\nLength Distribution:")
    bins = [0, 1024, 2048, 4096, 8192, 16384, 32768]
    for i in range(len(bins) - 1):
        count = sum(
            1 for length in sequence_lengths if bins[i] <= length < bins[i + 1]
        )
        pct = 100 * count / len(sequence_lengths)
        print(
            f"  {bins[i]:>6,} - {bins[i+1]:>6,} tokens:"
            f" {count:>5} ({pct:>5.1f}%)"
        )

    # Warn about very long sequences
    if too_long_examples:
        print(
            f"\n⚠️  WARNING: {len(too_long_examples)}"
            " examples exceed 8,192 tokens!"
        )
        print("  Longest examples:")
        for ex in sorted(
            too_long_examples, key=lambda x: x["length"], reverse=True
        )[:5]:
            print(
                f"  - claim_id={ex['claim_id']}, length={ex['length']:,} tokens"
            )

    # Recommend max_seq_length
    recommended_length = int(
        np.ceil(stats["p99_length"] / 1024) * 1024
    )  # Round up to nearest 1024
    print(f"\n💡 RECOMMENDED max_seq_length: {recommended_length:,} tokens")
    print("   (Covers 99% of examples)")

    # Plot distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(sequence_lengths, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(
        stats["mean_length"],
        color="red",
        linestyle="--",
        label=f'Mean: {stats["mean_length"]:.0f}',
    )
    plt.axvline(
        stats["p95_length"],
        color="orange",
        linestyle="--",
        label=f'95th: {stats["p95_length"]:.0f}',
    )
    plt.axvline(
        stats["p99_length"],
        color="green",
        linestyle="--",
        label=f'99th: {stats["p99_length"]:.0f}',
    )
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Sequence Length Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(
        sequence_lengths,
        bins=50,
        cumulative=True,
        density=True,
        edgecolor="black",
        alpha=0.7,
    )
    plt.axhline(0.95, color="orange", linestyle="--", label="95th percentile")
    plt.axhline(0.99, color="green", linestyle="--", label="99th percentile")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"sequence_length_analysis_{split}.png", dpi=300, bbox_inches="tight"
    )
    print(f"\n📊 Plot saved to: sequence_length_analysis_{split}.png")

    print("=" * 80 + "\n")

    return {
        "stats": stats,
        "sequence_lengths": sequence_lengths,
        "too_long_examples": too_long_examples,
        "recommended_max_seq_length": recommended_length,
    }


# Usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3-8B", trust_remote_code=True
    )

    # Analyze train set
    train_analysis = analyze_sequence_lengths(
        json_paths=[
            "data/perturbed_data/train_split/numeration_false.json",
            "data/perturbed_data/train_split/numeration_true.json",
        ],
        tokenizer=tokenizer,
        split="train",
        expand_perturbations=True,
        max_examples=None,  # Analyze subset for speed, or None for all
    )

    # # Analyze validation set
    # val_analysis = analyze_sequence_lengths(
    #     json_paths=["data/val.json"],
    #     tokenizer=tokenizer,
    #     split="validation",
    #     expand_perturbations=True
    # )
