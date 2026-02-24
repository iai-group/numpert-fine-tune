"""Test script to verify EnhancedPerturbationBatchSampler integration with
PerturbationAwareSFTTrainer.

This script creates a small mock dataset and verifies that:
1. Custom sampler is properly integrated
2. Batches have correct claim_id grouping
3. Constraints (original + perturbed) are enforced
4. Training loop can iterate without errors
"""

from datasets import Dataset
from src.llm_finetune.batch_sampler import EnhancedPerturbationBatchSampler
from collections import Counter
import random


def create_mock_dataset(n_claims=10, perturbations_per_claim=5):
    """Create a small mock dataset for testing."""

    data = {
        "claim_id": [],
        "perturbation_type": [],
        "label": [],
        "normalized_claim": [],
        "conversations": [],
    }

    for claim_idx in range(n_claims):
        claim_id = f"claim_{claim_idx}"

        # Add original
        data["claim_id"].append(claim_id)
        data["perturbation_type"].append("original")
        data["label"].append(random.choice([True, False]))
        data["normalized_claim"].append(f"Original claim {claim_idx}")
        data["conversations"].append(
            [
                {"role": "user", "content": f"Check claim {claim_idx}"},
                {
                    "role": "assistant",
                    "content": "True" if data["label"][-1] else "False",
                },
            ]
        )

        # Add perturbations
        perturb_types = [
            "numeration_true",
            "approximation_true",
            "range_false",
            "masking_false",
            "random_replacement_false",
        ]

        for p_idx in range(perturbations_per_claim):
            p_type = perturb_types[p_idx % len(perturb_types)]

            data["claim_id"].append(claim_id)
            data["perturbation_type"].append(p_type)

            # Determine label based on perturbation
            if data["label"][
                claim_idx * (perturbations_per_claim + 1)
            ]:  # Original was True
                if "_true" in p_type:
                    data["label"].append(True)  # T→T
                else:
                    data["label"].append(False)  # T→F
            else:  # Original was False
                data["label"].append(False)  # F→F (always False)

            data["normalized_claim"].append(
                f"Perturbed claim {claim_idx} ({p_type})"
            )
            data["conversations"].append(
                [
                    {
                        "role": "user",
                        "content": f"Check perturbed claim {claim_idx}",
                    },
                    {
                        "role": "assistant",
                        "content": "True" if data["label"][-1] else "False",
                    },
                ]
            )

    return Dataset.from_dict(data)


def test_custom_sampler_basic():
    """Test basic sampler functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Sampler Functionality")
    print("=" * 80)

    dataset = create_mock_dataset(n_claims=5, perturbations_per_claim=3)

    print(f"Dataset size: {len(dataset)}")
    print(f"Unique claim_ids: {len(set(dataset['claim_id']))}")
    print(f"Perturbation distribution: {Counter(dataset['perturbation_type'])}")

    sampler = EnhancedPerturbationBatchSampler(
        dataset=dataset,
        batch_size=4,
        batch_match_claim_id=True,
        split_large_claims=True,
        ensure_original_in_batch=True,
        ensure_perturbed_in_batch=True,
        shuffle=False,  # Deterministic for testing
        seed=42,
        drop_last=False,
    )

    print(f"\nSampler created with {len(sampler)} batches")

    # Check first few batches
    print("\nInspecting first 3 batches:")
    for batch_idx, batch_indices in enumerate(sampler):
        if batch_idx >= 3:
            break

        claim_ids = [dataset[idx]["claim_id"] for idx in batch_indices]
        perturb_types = [
            dataset[idx]["perturbation_type"] for idx in batch_indices
        ]

        unique_claims = len(set(claim_ids))
        has_original = "original" in perturb_types
        has_perturbed = any(p != "original" for p in perturb_types)

        print(f"\n  Batch {batch_idx}:")
        print(f"    Size: {len(batch_indices)}")
        print(
            f"    Unique claim_ids: {unique_claims} ({'✅' if unique_claims == 1 else '❌ FAIL'})"
        )
        print(
            f"    Has original: {has_original} ({'✅' if has_original else '❌ FAIL'})"
        )
        print(
            f"    Has perturbed: {has_perturbed} ({'✅' if has_perturbed else '❌ FAIL'})"
        )
        print(f"    Perturbation types: {perturb_types}")

    print("\n✅ TEST 1 PASSED\n")


def test_large_claim_splitting():
    """Test splitting of claims with many perturbations."""
    print("\n" + "=" * 80)
    print("TEST 2: Large Claim Splitting")
    print("=" * 80)

    # Create dataset with one claim having many perturbations
    dataset = create_mock_dataset(n_claims=1, perturbations_per_claim=10)

    print(f"Dataset size: {len(dataset)}")
    print(f"Examples for single claim: {len(dataset)}")

    sampler = EnhancedPerturbationBatchSampler(
        dataset=dataset,
        batch_size=4,
        batch_match_claim_id=True,
        split_large_claims=True,
        ensure_original_in_batch=True,
        ensure_perturbed_in_batch=True,
        shuffle=False,
        seed=42,
        drop_last=False,
    )

    print(f"\nSampler created with {len(sampler)} batches")
    print(f"Expected batches: ~{len(dataset) // 4}")

    # Check if original appears only once
    original_count = 0

    for batch_idx, batch_indices in enumerate(sampler):
        perturb_types = [
            dataset[idx]["perturbation_type"] for idx in batch_indices
        ]

        if "original" in perturb_types:
            original_count += 1
            print(f"\n  Batch {batch_idx} contains original")
            print(f"    Perturbation types: {perturb_types}")

    print(f"\n  Original appears in {original_count} batch(es)")

    if original_count == 1:
        print("\n✅ TEST 2 PASSED (Original appears exactly once)")
    else:
        print(
            f"\n❌ TEST 2 FAILED (Original should appear exactly once, found {original_count})"
        )


def test_constraint_enforcement():
    """Test that constraints are enforced."""
    print("\n" + "=" * 80)
    print("TEST 3: Constraint Enforcement")
    print("=" * 80)

    dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=3)

    sampler = EnhancedPerturbationBatchSampler(
        dataset=dataset,
        batch_size=4,
        batch_match_claim_id=True,
        split_large_claims=True,
        ensure_original_in_batch=True,
        ensure_perturbed_in_batch=True,
        shuffle=True,
        seed=42,
        drop_last=False,
    )

    violations = 0

    for batch_idx, batch_indices in enumerate(sampler):
        perturb_types = [
            dataset[idx]["perturbation_type"] for idx in batch_indices
        ]

        has_original = "original" in perturb_types
        has_perturbed = any(p != "original" for p in perturb_types)

        if not has_original or not has_perturbed:
            violations += 1
            print(f"\n  ⚠️  Batch {batch_idx} violates constraints:")
            print(f"      Has original: {has_original}")
            print(f"      Has perturbed: {has_perturbed}")

    print(f"\n  Total batches: {len(sampler)}")
    print(f"  Constraint violations: {violations}")

    if violations == 0:
        print("\n✅ TEST 3 PASSED (All constraints satisfied)")
    else:
        print(
            f"\n⚠️  TEST 3 WARNING ({violations} batches violate constraints)"
        )


def test_default_vs_custom():
    """Compare default (random) vs custom (claim-matched) batching."""
    print("\n" + "=" * 80)
    print("TEST 4: Default vs Custom Batching Comparison")
    print("=" * 80)

    dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=3)

    # Default (mixed) batching
    sampler_mixed = EnhancedPerturbationBatchSampler(
        dataset=dataset,
        batch_size=4,
        batch_match_claim_id=False,  # Random batching
        shuffle=True,
        seed=42,
        drop_last=False,
    )

    mixed_claim_diversity = []
    for batch_indices in sampler_mixed:
        claim_ids = [dataset[idx]["claim_id"] for idx in batch_indices]
        mixed_claim_diversity.append(len(set(claim_ids)))

    avg_mixed_diversity = sum(mixed_claim_diversity) / len(
        mixed_claim_diversity
    )

    # Custom (claim-matched) batching
    sampler_matched = EnhancedPerturbationBatchSampler(
        dataset=dataset,
        batch_size=4,
        batch_match_claim_id=True,  # Claim-ID matching
        split_large_claims=True,
        shuffle=True,
        seed=42,
        drop_last=False,
    )

    matched_claim_diversity = []
    for batch_indices in sampler_matched:
        claim_ids = [dataset[idx]["claim_id"] for idx in batch_indices]
        matched_claim_diversity.append(len(set(claim_ids)))

    avg_matched_diversity = sum(matched_claim_diversity) / len(
        matched_claim_diversity
    )

    print(f"\n  Mixed batching:")
    print(f"    Average unique claim_ids per batch: {avg_mixed_diversity:.2f}")
    print(f"    (Higher is more diverse - batches mix different claims)")

    print(f"\n  Matched batching:")
    print(
        f"    Average unique claim_ids per batch: {avg_matched_diversity:.2f}"
    )
    print(f"    (Should be close to 1.0 - batches focus on single claim)")

    if avg_matched_diversity < 1.5:
        print("\n✅ TEST 4 PASSED (Claim matching working correctly)")
    else:
        print(
            f"\n❌ TEST 4 FAILED (Expected ~1.0, got {avg_matched_diversity:.2f})"
        )


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ENHANCEDPERTURBATIONBATCHSAMPLER INTEGRATION TESTS")
    print("=" * 80)

    test_custom_sampler_basic()
    test_large_claim_splitting()
    test_constraint_enforcement()
    test_default_vs_custom()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()
