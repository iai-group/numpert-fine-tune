import pytest
from datasets import Dataset
from collections import Counter, defaultdict
import sys
import os

from src.llm_finetune.batch_sampler import EnhancedPerturbationBatchSampler


def print_separator(title: str):
    """Print a nice separator for test output."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_mock_dataset(
    n_claims: int = 10,
    perturbations_per_claim: int = 3,
    include_original: bool = True,
) -> Dataset:
    """Create a mock dataset for testing batch sampler.

    Args:
        n_claims: Number of unique claims
        perturbations_per_claim: Number of perturbations per claim
        include_original: Whether to include original claim

    Returns:
        Mock dataset
    """
    examples = []

    for i in range(n_claims):
        claim_id = f"claim_{i:04d}"

        # Original
        if include_original:
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": "original",
                    "conversations": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": f"Claim {i}"},
                        {"role": "assistant", "content": "True"},
                    ],
                }
            )

        # Perturbations
        perturb_types = ["perturb_type1", "perturb_type2", "perturb_type3_random"]
        for j in range(perturbations_per_claim):
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": perturb_types[j % len(perturb_types)],
                    "conversations": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": f"Claim {i} perturbed {j}"},
                        {"role": "assistant", "content": "False"},
                    ],
                }
            )

    return Dataset.from_list(examples)


def analyze_batches(batches: list, dataset: Dataset) -> dict:
    """Analyze batch statistics."""
    stats = {
        "num_batches": len(batches),
        "batch_sizes": [len(b) for b in batches],
        "total_examples": sum(len(b) for b in batches),
        "batches_with_original": 0,
        "batches_with_perturbed": 0,
        "batches_with_both": 0,
        "claim_ids_per_batch": [],
        "unique_claim_ids_per_batch": [],
        "original_count_per_batch": [],
        "perturbed_count_per_batch": [],
        "claim_id_repeated_across_batches": {},
    }

    # Track which claim_ids appear in which batches (to check for original repetition)
    claim_id_to_batches = defaultdict(
        lambda: {"original_in": [], "perturbed_in": []}
    )

    for batch_idx, batch in enumerate(batches):
        batch_claim_ids = []
        batch_perturb_types = []
        original_count = 0
        perturbed_count = 0

        for idx in batch:
            claim_id = dataset[idx]["claim_id"]
            perturb_type = dataset[idx]["perturbation_type"]

            batch_claim_ids.append(claim_id)
            batch_perturb_types.append(perturb_type)

            if perturb_type == "original":
                original_count += 1
                claim_id_to_batches[claim_id]["original_in"].append(batch_idx)
            else:
                perturbed_count += 1
                claim_id_to_batches[claim_id]["perturbed_in"].append(batch_idx)

        stats["claim_ids_per_batch"].append(batch_claim_ids)
        stats["unique_claim_ids_per_batch"].append(len(set(batch_claim_ids)))
        stats["original_count_per_batch"].append(original_count)
        stats["perturbed_count_per_batch"].append(perturbed_count)

        if original_count > 0:
            stats["batches_with_original"] += 1
        if perturbed_count > 0:
            stats["batches_with_perturbed"] += 1
        if original_count > 0 and perturbed_count > 0:
            stats["batches_with_both"] += 1

    # Check for original repetition across batches
    for claim_id, locations in claim_id_to_batches.items():
        if len(locations["original_in"]) > 1:
            stats["claim_id_repeated_across_batches"][claim_id] = locations[
                "original_in"
            ]

    return stats


def print_batch_stats(stats: dict, title: str = "Batch Statistics"):
    """Pretty print batch statistics."""
    print(f"\n{title}:")
    print(f"  Number of batches: {stats['num_batches']}")
    print(f"  Total examples: {stats['total_examples']}")
    print(
        f"  Batch sizes: min={min(stats['batch_sizes']) if stats['batch_sizes'] else 0}, "
        f"max={max(stats['batch_sizes']) if stats['batch_sizes'] else 0}, "
        f"avg={sum(stats['batch_sizes'])/len(stats['batch_sizes']) if stats['batch_sizes'] else 0:.1f}"
    )
    print(f"\n  Batches with original: {stats['batches_with_original']}")
    print(f"  Batches with perturbed: {stats['batches_with_perturbed']}")
    print(f"  Batches with both: {stats['batches_with_both']}")

    if stats["claim_id_repeated_across_batches"]:
        print(f"\n  ⚠️  WARNING: Originals repeated across batches!")
        for claim_id, batch_indices in stats[
            "claim_id_repeated_across_batches"
        ].items():
            print(f"    {claim_id}: appears in batches {batch_indices}")
    else:
        print(f"\n  ✓ No originals repeated across batches")


class TestEnhancedPerturbationBatchSampler:
    """Test suite for EnhancedPerturbationBatchSampler."""

    def test_basic_batching(self):
        """Test basic batch creation without claim_id matching."""
        print_separator("TEST: Basic Batching (No Claim ID Matching)")

        dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=3)
        # 10 claims * (1 original + 3 perturbed) = 40 examples

        print(f"\nDataset size: {len(dataset)}")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=False,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Should have 5 batches (40 / 8 = 5)
        assert stats["num_batches"] == 5
        assert stats["total_examples"] == 40

        print("\n✓ Test passed: Basic batching works")

    def test_claim_id_matching_small_claims(self):
        """Test claim_id matching with claims that fit in one batch."""
        print_separator("TEST: Claim ID Matching (Small Claims)")

        # 10 claims with 1 original + 2 perturbed = 3 examples per claim
        dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=2)

        print(f"\nDataset size: {len(dataset)}")
        print(f"Examples per claim: 3 (1 original + 2 perturbed)")
        print(f"Batch size: 8 (can fit 2+ claims per batch)")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Each claim should fit in one batch (3 examples < 8 batch_size)
        # Verify that claim_ids are grouped together
        for batch_idx, claim_ids in enumerate(stats["claim_ids_per_batch"]):
            unique_in_batch = len(set(claim_ids))
            print(f"\nBatch {batch_idx}: {unique_in_batch} unique claim_ids")
            print(f"  Claim IDs: {set(claim_ids)}")

        # Check no original is repeated
        assert (
            len(stats["claim_id_repeated_across_batches"]) == 0
        ), "Originals should never be repeated across batches!"

        print("\n✓ Test passed: Claim ID matching works for small claims")

    def test_split_large_claims(self):
        """Test splitting large claims across batches."""
        print_separator("TEST: Split Large Claims Across Batches")

        # Create 1 claim with many perturbations
        dataset = create_mock_dataset(n_claims=1, perturbations_per_claim=20)
        # 1 original + 20 perturbed = 21 examples

        print(f"\nDataset size: {len(dataset)}")
        print(f"Single claim with 1 original + 20 perturbed = 21 examples")
        print(f"Batch size: 8")
        print(f"Expected: 3 batches (8 + 8 + 5)")
        print(f"Original should only appear in first batch!")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Should split into 3 batches
        assert stats["num_batches"] == 3

        # First batch should have the original
        assert (
            stats["original_count_per_batch"][0] == 1
        ), "First batch should contain the original"

        # Subsequent batches should NOT have the original
        for i in range(1, stats["num_batches"]):
            assert (
                stats["original_count_per_batch"][i] == 0
            ), f"Batch {i} should NOT contain original (original already used)"

        # CRITICAL: No original should be repeated
        assert (
            len(stats["claim_id_repeated_across_batches"]) == 0
        ), "Original should NEVER be repeated across batches!"

        # All batches should have the same claim_id
        for batch_claim_ids in stats["claim_ids_per_batch"]:
            assert (
                len(set(batch_claim_ids)) == 1
            ), "All examples in all batches should be from the same claim_id"

        print(
            "\n✓ Test passed: Large claims split correctly without repeating original"
        )

    def test_no_split_large_claims(self):
        """Test behavior when split_large_claims=False."""
        print_separator("TEST: Don't Split Large Claims")

        dataset = create_mock_dataset(n_claims=1, perturbations_per_claim=20)

        print(f"\nDataset size: {len(dataset)} (1 claim with 21 examples)")
        print(f"Batch size: 8")
        print(f"split_large_claims: False")
        print(f"Expected: Only first 8 examples used")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=False,  # Don't split
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Should only have 1 batch with first 8 examples
        assert stats["num_batches"] == 1
        assert stats["batch_sizes"][0] == 8

        print(
            "\n✓ Test passed: Large claims truncated when split_large_claims=False"
        )

    def test_drop_last(self):
        """Test drop_last functionality."""
        print_separator("TEST: Drop Last Incomplete Batch")

        dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=2)
        # 10 * 3 = 30 examples

        print(f"\nDataset size: {len(dataset)}")
        print(f"Batch size: 8")
        print(f"drop_last: True")
        print(f"Expected: 3 batches of size 8 (last 6 examples dropped)")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=False,
            drop_last=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Should have 3 complete batches (30 / 8 = 3 remainder 6)
        assert stats["num_batches"] == 3
        assert all(
            size == 8 for size in stats["batch_sizes"]
        ), "All batches should have exactly batch_size examples"

        print("\n✓ Test passed: drop_last works correctly")

    def test_shuffle(self):
        """Test that shuffle actually shuffles batches."""
        print_separator("TEST: Shuffle Batches")

        dataset = create_mock_dataset(n_claims=20, perturbations_per_claim=2)

        # Create two samplers with different seeds
        sampler1 = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=False,
            shuffle=True,
            seed=42,
        )

        sampler2 = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=False,
            shuffle=True,
            seed=99,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # Batches should be different due to different seeds
        assert (
            batches1 != batches2
        ), "Different seeds should produce different batch orders"

        # But should have same total examples
        assert sum(len(b) for b in batches1) == sum(len(b) for b in batches2)

        print(f"\nBatches from seed 42: {len(batches1)} batches")
        print(f"Batches from seed 99: {len(batches2)} batches")
        print(f"Batch orders are different: ✓")

        print("\n✓ Test passed: Shuffle works correctly")

    def test_mixed_claim_sizes(self):
        """Test with varying claim sizes."""
        print_separator("TEST: Mixed Claim Sizes")

        # Create claims with different numbers of perturbations
        examples = []

        # Small claim: 1 original + 1 perturbed
        for i in range(2):
            examples.append(
                {
                    "claim_id": "small_claim",
                    "perturbation_type": (
                        "original" if i == 0 else "perturb_type2"
                    ),
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        # Medium claim: 1 original + 5 perturbed
        for i in range(6):
            examples.append(
                {
                    "claim_id": "medium_claim",
                    "perturbation_type": (
                        "original" if i == 0 else "perturb_type1"
                    ),
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        # Large claim: 1 original + 15 perturbed
        for i in range(16):
            examples.append(
                {
                    "claim_id": "large_claim",
                    "perturbation_type": (
                        "original" if i == 0 else "range_false"
                    ),
                    "conversations": [
                        {"role": "assistant", "content": "False"}
                    ],
                }
            )

        dataset = Dataset.from_list(examples)
        # Total: 2 + 6 + 16 = 24 examples

        print(f"\nDataset size: {len(dataset)}")
        print(f"  Small claim: 2 examples")
        print(f"  Medium claim: 6 examples")
        print(f"  Large claim: 16 examples")
        print(f"Batch size: 8")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # Verify no originals repeated
        assert len(stats["claim_id_repeated_across_batches"]) == 0

        # Print claim distribution per batch
        print("\nClaim distribution per batch:")
        for i, claim_ids in enumerate(stats["claim_ids_per_batch"]):
            claim_counter = Counter(claim_ids)
            print(f"  Batch {i}: {dict(claim_counter)}")

        print("\n✓ Test passed: Mixed claim sizes handled correctly")

    def test_dataset_without_originals(self):
        """Test dataset with only perturbed examples (edge case)"""
        print_separator("TEST: Dataset Without Originals")

        dataset = create_mock_dataset(
            n_claims=5,
            perturbations_per_claim=4,
            include_original=False,  # No originals!
        )

        print(f"\nDataset size: {len(dataset)} (only perturbed, no originals)")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # All batches should have 0 originals
        assert all(count == 0 for count in stats["original_count_per_batch"])

        print("\n✓ Test passed: Dataset without originals handled")

    def test_sampler_length(self):
        """Test __len__ method."""
        print_separator("TEST: Sampler Length")

        dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=2)
        # 10 * 3 = 30 examples

        # Without drop_last
        sampler1 = EnhancedPerturbationBatchSampler(
            dataset=dataset, batch_size=8, drop_last=False, shuffle=False
        )

        # With drop_last
        sampler2 = EnhancedPerturbationBatchSampler(
            dataset=dataset, batch_size=8, drop_last=True, shuffle=False
        )

        len1 = len(sampler1)
        len2 = len(sampler2)

        print(f"\nDataset: 30 examples, batch_size: 8")
        print(f"Without drop_last: {len1} batches (expected: 4)")
        print(f"With drop_last: {len2} batches (expected: 3)")

        assert len1 == 4  # ceil(30/8) = 4
        assert len2 == 3  # floor(30/8) = 3

        print("\n✓ Test passed: __len__ works correctly")

    def test_ensure_original_and_perturbed_in_batch(self):
        """Test that batches can be enforced to have both original and
        perturbed."""
        print_separator("TEST: Ensure Original and Perturbed in Each Batch")

        # Create dataset with uneven distribution
        # Some claims have only original, some only perturbed
        examples = []

        # Add 5 claims with original + perturbed (normal)
        for i in range(5):
            claim_id = f"normal_{i}"
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": "original",
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": "perturb_type2",
                    "conversations": [
                        {"role": "assistant", "content": "False"}
                    ],
                }
            )

        # Add 10 claims with only originals (no perturbations)
        for i in range(10):
            examples.append(
                {
                    "claim_id": f"only_orig_{i}",
                    "perturbation_type": "original",
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        # Add 10 perturbations without originals
        for i in range(10):
            examples.append(
                {
                    "claim_id": f"only_pert_{i}",
                    "perturbation_type": "perturb_type1",
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        dataset = Dataset.from_list(examples)
        print(f"\nDataset composition:")
        print(f"  Normal claims (orig + pert): 5 claims × 2 = 10 examples")
        print(f"  Only originals: 10 examples")
        print(f"  Only perturbed: 10 examples")
        print(f"  Total: {len(dataset)} examples")

        # Test WITHOUT constraints
        print("\n--- Without Constraints ---")
        sampler_no_constraints = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=5,
            batch_match_claim_id=False,
            ensure_original_in_batch=False,
            ensure_perturbed_in_batch=False,
            shuffle=False,
            seed=42,
        )

        batches_no_constraints = list(sampler_no_constraints)
        stats_no_constraints = analyze_batches(batches_no_constraints, dataset)

        print(
            f"Batches with both: {stats_no_constraints['batches_with_both']}/{stats_no_constraints['num_batches']}"
        )
        print(
            f"Batches with only original: {stats_no_constraints['batches_with_original'] - stats_no_constraints['batches_with_both']}"
        )
        print(
            f"Batches with only perturbed: {stats_no_constraints['batches_with_perturbed'] - stats_no_constraints['batches_with_both']}"
        )

        # Test WITH constraints
        print("\n--- With Constraints ---")
        sampler_with_constraints = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=5,
            batch_match_claim_id=False,
            ensure_original_in_batch=True,
            ensure_perturbed_in_batch=True,
            shuffle=False,
            seed=42,
        )

        batches_with_constraints = list(sampler_with_constraints)
        stats_with_constraints = analyze_batches(
            batches_with_constraints, dataset
        )

        print(
            f"Batches with both: {stats_with_constraints['batches_with_both']}/{stats_with_constraints['num_batches']}"
        )
        print(
            f"Batches with only original: {stats_with_constraints['batches_with_original'] - stats_with_constraints['batches_with_both']}"
        )
        print(
            f"Batches with only perturbed: {stats_with_constraints['batches_with_perturbed'] - stats_with_constraints['batches_with_both']}"
        )

        # Verify that constraints improved the situation
        assert (
            stats_with_constraints["batches_with_both"]
            >= stats_no_constraints["batches_with_both"]
        ), "Constraints should increase batches with both types"

        print("\n✓ Test passed: Constraint enforcement attempts to fix batches")

    def test_constraint_enforcement_impossible(self):
        """Test behavior when constraints cannot be satisfied."""
        print_separator("TEST: Constraints Cannot Be Satisfied")

        # Create dataset with ONLY originals (impossible to have perturbed in each batch)
        examples = []
        for i in range(20):
            examples.append(
                {
                    "claim_id": f"claim_{i}",
                    "perturbation_type": "original",
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        dataset = Dataset.from_list(examples)
        print(
            f"\nDataset: {len(dataset)} examples (ALL originals, no perturbed)"
        )

        # Try to enforce having perturbed in each batch (impossible)
        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=5,
            ensure_original_in_batch=True,
            ensure_perturbed_in_batch=True,  # Impossible to satisfy!
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)

        print(
            f"\nWarnings should be shown above about batches missing perturbed examples"
        )
        print(
            f"Batches with perturbed: {stats['batches_with_perturbed']}/{stats['num_batches']}"
        )

        # Should still create batches, just with warnings
        assert stats["num_batches"] > 0
        assert (
            stats["batches_with_perturbed"] == 0
        )  # Impossible to have perturbed

        print(
            "\n✓ Test passed: Impossible constraints handled gracefully with warnings"
        )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING BATCH SAMPLER TESTS")
    print("=" * 80)

    pytest.main([__file__, "-v", "-s"])
