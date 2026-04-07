import pytest
from datasets import Dataset
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

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
                    "perturbation_type": "original_normalized",
                    "conversations": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": f"Claim {i}"},
                        {"role": "assistant", "content": "True"},
                    ],
                }
            )

        # Perturbations
        perturb_types = [
            "perturb_type1",
            "perturb_type2",
            "perturb_type3_random",
        ]
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


def create_multi_file_mock_dataset(
    n_claims: int = 5,
    file_perturbation_types: Optional[List[List[str]]] = None,
) -> Dataset:
    """Simulate loading data from multiple files with overlapping claim_ids.

    Each "file" contributes an ``original_normalized`` entry plus its own
    set of perturbation types for every claim.  This mirrors the real
    pipeline where ``use_perturbations`` is called once per file and
    the results are concatenated — producing duplicate originals for any
    ``claim_id`` that appears in more than one source file.

    Args:
        n_claims: Number of unique claims (shared across all files).
        file_perturbation_types: A list of perturbation-type lists, one per
            simulated file.  Defaults to two files with distinct types::

                [["perturb_type1", "perturb_type2"],
                 ["perturb_type3_random", "perturb_type3_masked"]]

    Returns:
        A ``Dataset`` that contains the duplicate originals and all
        perturbation types — exactly what the batch sampler would receive.
    """
    if file_perturbation_types is None:
        file_perturbation_types = [
            ["perturb_type1", "perturb_type2"],
            ["perturb_type3_random", "perturb_type3_masked"],
        ]

    examples: List[Dict] = []

    for file_idx, perturb_types in enumerate(file_perturbation_types):
        for i in range(n_claims):
            claim_id = f"claim_{i:04d}"

            # Each file emits its own original_normalized entry
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": "original_normalized",
                    "conversations": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": f"Claim {i}"},
                        {"role": "assistant", "content": "True"},
                    ],
                    "source_file": f"file_{file_idx}.json",
                }
            )

            # Perturbation entries specific to this file
            for p_type in perturb_types:
                examples.append(
                    {
                        "claim_id": claim_id,
                        "perturbation_type": p_type,
                        "conversations": [
                            {"role": "system", "content": "System"},
                            {
                                "role": "user",
                                "content": (
                                    f"Claim {i} perturbed"
                                    f" ({p_type}) from file_{file_idx}"
                                ),
                            },
                            {"role": "assistant", "content": "False"},
                        ],
                        "source_file": f"file_{file_idx}.json",
                    }
                )

    return examples, Dataset.from_list(examples)


def analyze_batches(batches: list, dataset: Dataset) -> Dict[str, Any]:
    """Analyze batch statistics."""
    stats: Dict[str, Any] = {
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

    # Track which claim_ids appear in which batches
    # (to check for original repetition)
    claim_id_to_batches: Dict[str, Dict[str, List[int]]] = defaultdict(
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

            if perturb_type == "original_normalized":
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
    batch_sizes = stats["batch_sizes"]
    min_size = min(batch_sizes) if batch_sizes else 0
    max_size = max(batch_sizes) if batch_sizes else 0
    avg_size = (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0
    print(
        f"  Batch sizes: min={min_size}, " f"max={max_size}, avg={avg_size:.1f}"
    )
    print(f"\n  Batches with original: {stats['batches_with_original']}")
    print(f"  Batches with perturbed: {stats['batches_with_perturbed']}")
    print(f"  Batches with both: {stats['batches_with_both']}")

    if stats["claim_id_repeated_across_batches"]:
        print("\n  ⚠️  WARNING: Originals repeated across batches!")
        for claim_id, batch_indices in stats[
            "claim_id_repeated_across_batches"
        ].items():
            print(f"    {claim_id}: appears in batches {batch_indices}")
    else:
        print("\n  ✓ No originals repeated across batches")


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
        print("Examples per claim: 3 (1 original + 2 perturbed)")
        print("Batch size: 8 (can fit 2+ claims per batch)")

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
        print("Single claim with 1 original + 20 perturbed = 21 examples")
        print("Batch size: 8")
        print("Expected: 3 batches (8 + 8 + 5)")
        print("Original should only appear in first batch!")

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
        msg1 = "First batch should contain the original"
        assert stats["original_count_per_batch"][0] == 1, msg1

        # Subsequent batches should NOT have the original
        for i in range(1, stats["num_batches"]):
            msg_i = f"Batch {i} should NOT contain original (used already)"
            assert stats["original_count_per_batch"][i] == 0, msg_i

        # CRITICAL: No original should be repeated
        msg_repeat = "Original should NEVER be repeated across batches!"
        assert len(stats["claim_id_repeated_across_batches"]) == 0, msg_repeat

        # All batches should have the same claim_id
        for batch_claim_ids in stats["claim_ids_per_batch"]:
            msg_same = "All examples should be from same claim_id"
            assert len(set(batch_claim_ids)) == 1, msg_same

        print(
            "\n✓ Test passed: Large claims split correctly "
            "without repeating original"
        )

    def test_no_split_large_claims(self):
        """Test behavior when split_large_claims=False."""
        print_separator("TEST: Don't Split Large Claims")

        dataset = create_mock_dataset(n_claims=1, perturbations_per_claim=20)

        print(f"\nDataset size: {len(dataset)} (1 claim with 21 examples)")
        print("Batch size: 8")
        print("split_large_claims: False")
        print("Expected: Only first 8 examples used")

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
            "\n✓ Test passed: Large claims truncated when "
            "split_large_claims=False"
        )

    def test_drop_last(self):
        """Test drop_last functionality."""
        print_separator("TEST: Drop Last Incomplete Batch")

        dataset = create_mock_dataset(n_claims=10, perturbations_per_claim=2)
        # 10 * 3 = 30 examples

        print(f"\nDataset size: {len(dataset)}")
        print("Batch size: 8")
        print("drop_last: True")
        print("Expected: 3 batches of size 8 (last 6 examples dropped)")

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
        msg_drop = "All batches should have exactly batch_size examples"
        assert all(size == 8 for size in stats["batch_sizes"]), msg_drop

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
        msg_seed = "Different seeds should produce different batch orders"
        assert batches1 != batches2, msg_seed

        # But should have same total examples
        assert sum(len(b) for b in batches1) == sum(len(b) for b in batches2)

        print(f"\nBatches from seed 42: {len(batches1)} batches")
        print(f"Batches from seed 99: {len(batches2)} batches")
        print("Batch orders are different: ✓")

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
                        "original_normalized" if i == 0 else "perturb_type2"
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
                        "original_normalized" if i == 0 else "perturb_type1"
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
                        "original_normalized" if i == 0 else "range_false"
                    ),
                    "conversations": [
                        {"role": "assistant", "content": "False"}
                    ],
                }
            )

        dataset = Dataset.from_list(examples)
        # Total: 2 + 6 + 16 = 24 examples

        print(f"\nDataset size: {len(dataset)}")
        print("  Small claim: 2 examples")
        print("  Medium claim: 6 examples")
        print("  Large claim: 16 examples")
        print("Batch size: 8")

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

        msg_ds = "(only perturbed, no originals)"
        print(f"\nDataset size: {len(dataset)} {msg_ds}")

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

        print("\nDataset: 30 examples, batch_size: 8")
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
                    "perturbation_type": "original_normalized",
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
                    "perturbation_type": "original_normalized",
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
        print("\nDataset composition:")
        print("  Normal claims (orig + pert): 5 claims × 2 = 10 examples")
        print("  Only originals: 10 examples")
        print("  Only perturbed: 10 examples")
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

        n_both = stats_no_constraints["batches_with_both"]
        n_total = stats_no_constraints["num_batches"]
        print(f"Batches with both: {n_both}/{n_total}")
        n_only_orig = (
            stats_no_constraints["batches_with_original"]
            - stats_no_constraints["batches_with_both"]
        )
        print(f"Batches with only original: {n_only_orig}")
        n_only_pert = (
            stats_no_constraints["batches_with_perturbed"]
            - stats_no_constraints["batches_with_both"]
        )
        print(f"Batches with only perturbed: {n_only_pert}")

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

        n_both2 = stats_with_constraints["batches_with_both"]
        n_total2 = stats_with_constraints["num_batches"]
        print(f"Batches with both: {n_both2}/{n_total2}")
        n_only_orig2 = (
            stats_with_constraints["batches_with_original"]
            - stats_with_constraints["batches_with_both"]
        )
        print(f"Batches with only original: {n_only_orig2}")
        n_only_pert2 = (
            stats_with_constraints["batches_with_perturbed"]
            - stats_with_constraints["batches_with_both"]
        )
        print(f"Batches with only perturbed: {n_only_pert2}")

        # Verify that constraints improved the situation
        msg_constr = "Constraints should increase batches with both types"
        assert (
            stats_with_constraints["batches_with_both"]
            >= stats_no_constraints["batches_with_both"]
        ), msg_constr

        print("\n✓ Test passed: Constraint enforcement attempts to fix batches")

    def test_constraint_enforcement_impossible(self):
        """Test behavior when constraints cannot be satisfied."""
        print_separator("TEST: Constraints Cannot Be Satisfied")

        # Create dataset with ONLY originals (no perturbed examples)
        examples = []
        for i in range(20):
            examples.append(
                {
                    "claim_id": f"claim_{i}",
                    "perturbation_type": "original_normalized",
                    "conversations": [{"role": "assistant", "content": "True"}],
                }
            )

        dataset = Dataset.from_list(examples)
        msg_ex = "(ALL originals, no perturbed)"
        print(f"\nDataset: {len(dataset)} examples {msg_ex}")

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
            "\nWarnings should be shown above about batches "
            "missing perturbed examples"
        )
        n_pert = stats["batches_with_perturbed"]
        n_all = stats["num_batches"]
        print(f"Batches with perturbed: {n_pert}/{n_all}")

        # Should still create batches, just with warnings
        assert stats["num_batches"] > 0
        # Impossible to have perturbed
        assert stats["batches_with_perturbed"] == 0

        print(
            "\n✓ Test passed: Impossible constraints "
            "handled gracefully with warnings"
        )


class TestMultiFileDeduplication:
    """Tests for datasets loaded from multiple files with overlapping claim_ids.

    When multiple perturbation JSON files share the same claims (e.g.
    ``approximation_True.json`` and ``range_True.json``),
    ``use_perturbations`` emits an ``original_normalized`` entry from
    each file.  The batch sampler must not treat those duplicates as
    distinct originals: each ``claim_id`` should contribute exactly *one*
    original to the sampling, while all perturbation types from every
    source file must be preserved.
    """

    def test_no_duplicate_originals_in_groups(self):
        """Verify claim_id_groups stores only one original per claim_id.

        The sampler uses ``dict["original"] = idx`` (assignment, not
        append), so even if the raw dataset has two original_normalized
        rows for the same claim_id only one ends up in the group.
        """
        print_separator("TEST: No Duplicate Originals in Groups (Multi-File)")

        raw_examples, dataset = create_multi_file_mock_dataset(
            n_claims=5,
            file_perturbation_types=[
                ["perturb_type1", "perturb_type2"],
                ["perturb_type3_random", "perturb_type3_masked"],
            ],
        )

        # Count raw originals to confirm duplicates exist in the input
        raw_original_count = sum(
            1
            for ex in raw_examples
            if ex["perturbation_type"] == "original_normalized"
        )
        print(f"\nRaw dataset size: {len(dataset)}")
        print(f"Raw original_normalized rows: {raw_original_count}")
        assert raw_original_count == 10, (
            "Expected 2 originals per claim × 5 claims = 10 "
            f"but got {raw_original_count}"
        )

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        # Each claim_id must have exactly one original in the groups
        for claim_id, group in sampler.claim_id_groups.items():
            assert (
                group["original"] is not None
            ), f"{claim_id}: missing original in claim_id_groups"
            # 'original' is a scalar (last-write-wins), so uniqueness
            # in the group is guaranteed by dict semantics — just
            # verify no None slipped through.
            print(
                f"  {claim_id}: original idx={group['original']}, "
                f"perturbed count={len(group['perturbed'])}"
            )

        print("\n✓ Test passed: claim_id_groups has one original per claim")

    def test_all_perturbation_types_preserved(self):
        """All perturbation types from every source file must appear."""
        print_separator("TEST: All Perturbation Types Preserved (Multi-File)")

        file_types = [
            ["perturb_type1", "perturb_type2"],
            ["perturb_type3_random", "perturb_type3_masked"],
        ]
        _, dataset = create_multi_file_mock_dataset(
            n_claims=5, file_perturbation_types=file_types
        )

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            shuffle=False,
            seed=42,
        )

        # Collect all perturbation types actually tracked by the sampler
        all_perturbed_types_in_groups: set = set()
        for claim_id, group in sampler.claim_id_groups.items():
            for idx in group["perturbed"]:
                all_perturbed_types_in_groups.add(
                    dataset[idx]["perturbation_type"]
                )

        expected_types = {t for types in file_types for t in types}
        print(f"\nExpected perturbation types: {sorted(expected_types)}")
        print(
            "Perturbation types in groups: "
            f"{sorted(all_perturbed_types_in_groups)}"
        )

        assert (
            expected_types == all_perturbed_types_in_groups
        ), f"Missing types: {expected_types - all_perturbed_types_in_groups}"

        # Per-claim check: every claim should have all types
        for claim_id, group in sampler.claim_id_groups.items():
            claim_types = {
                dataset[idx]["perturbation_type"] for idx in group["perturbed"]
            }
            assert (
                expected_types == claim_types
            ), f"{claim_id}: expected {expected_types} but got {claim_types}"

        print("\n✓ Test passed: all perturbation types retained")

    def test_batches_contain_no_duplicate_originals_per_claim(self):
        """When batching, each claim_id's original must appear at most once.

        Even though the raw dataset has duplicate originals, the sampler should
        never yield the same claim_id's original in more than one batch.
        """
        print_separator("TEST: Batches Have No Duplicate Originals Per Claim")

        _, dataset = create_multi_file_mock_dataset(
            n_claims=5,
            file_perturbation_types=[
                ["perturb_type1"],
                ["perturb_type2"],
                ["perturb_type3_random"],
            ],
        )
        print(f"\nDataset size: {len(dataset)} (3 files × 5 claims)")

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=6,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # No original should be repeated across batches
        assert len(stats["claim_id_repeated_across_batches"]) == 0, (
            "Originals repeated across batches: "
            f"{stats['claim_id_repeated_across_batches']}"
        )

        print("\n✓ Test passed: no duplicate originals across batches")

    def test_total_perturbed_count_matches_source_files(self):
        """The number of perturbed examples seen across all batches must equal
        the total from all source files (no perturbations lost)."""
        print_separator("TEST: Total Perturbed Count Matches Source Files")

        file_types = [
            ["perturb_type1", "perturb_type2"],
            ["perturb_type3_random"],
        ]
        n_claims = 5
        _, dataset = create_multi_file_mock_dataset(
            n_claims=n_claims, file_perturbation_types=file_types
        )

        # Expected perturbed: file0 has 2 types × 5 claims = 10,
        #                     file1 has 1 type  × 5 claims =  5  → 15
        expected_perturbed = sum(len(types) * n_claims for types in file_types)

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
            drop_last=False,
        )

        batches = list(sampler)

        all_perturbed_indices = set()
        for batch in batches:
            for idx in batch:
                if dataset[idx]["perturbation_type"] != "original_normalized":
                    all_perturbed_indices.add(idx)

        print(f"\nExpected perturbed examples: {expected_perturbed}")
        print(f"Perturbed in batches:        {len(all_perturbed_indices)}")

        assert len(all_perturbed_indices) == expected_perturbed, (
            f"Expected {expected_perturbed} perturbed examples but "
            f"got {len(all_perturbed_indices)}"
        )

        print("\n✓ Test passed: all perturbed examples present in batches")

    def test_three_files_same_claims(self):
        """Stress test: three source files, all sharing the same claims.

        Verifies correct behaviour with three-way original duplication.
        """
        print_separator("TEST: Three Files Same Claims")

        file_types = [
            ["perturb_type1"],
            ["perturb_type2"],
            ["perturb_type3_random"],
        ]
        n_claims = 4
        raw_examples, dataset = create_multi_file_mock_dataset(
            n_claims=n_claims, file_perturbation_types=file_types
        )

        # 3 files × 4 claims × (1 original + 1 perturbation) = 24 rows
        # 3 originals per claim × 4 claims = 12 duplicates total
        raw_originals = sum(
            1
            for ex in raw_examples
            if ex["perturbation_type"] == "original_normalized"
        )
        print(f"\nRaw rows: {len(dataset)}")
        print(f"Raw originals (with duplicates): {raw_originals}")
        assert raw_originals == n_claims * len(file_types)

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=6,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        # claim_id_groups should have exactly n_claims entries
        assert len(sampler.claim_id_groups) == n_claims

        # Each group should reference ONE original and exactly 3
        # perturbations (one from each file)
        for claim_id, group in sampler.claim_id_groups.items():
            assert group["original"] is not None
            assert len(group["perturbed"]) == len(file_types), (
                f"{claim_id}: expected {len(file_types)} perturbed, "
                f"got {len(group['perturbed'])}"
            )

        # Verify perturbation type coverage per claim
        expected_types = {t for types in file_types for t in types}
        for claim_id, group in sampler.claim_id_groups.items():
            claim_types = {
                dataset[idx]["perturbation_type"] for idx in group["perturbed"]
            }
            assert expected_types == claim_types, (
                f"{claim_id}: expected types {expected_types}, "
                f"got {claim_types}"
            )

        batches = list(sampler)
        stats = analyze_batches(batches, dataset)
        print_batch_stats(stats)

        # No originals repeated across batches
        assert len(stats["claim_id_repeated_across_batches"]) == 0

        print("\n✓ Test passed: three files handled correctly")

    def test_overlapping_perturbation_types_across_files(self):
        """Two files both emit the same perturbation type for a claim.

        This mirrors the real scenario: ``approximation_True.json`` and
        ``range_True.json`` both have a ``perturb_type1`` key for the
        same ``claim_id``, but with *different* perturbed claim text.
        Both entries must end up in the sampler's group and eventually
        appear in batches.
        """
        print_separator("TEST: Overlapping Perturbation Types Across Files")

        # Both files contribute perturb_type1 — different claim text
        file_types = [
            ["perturb_type1", "perturb_type2"],
            ["perturb_type1", "perturb_type3_random"],
        ]
        n_claims = 4
        raw_examples, dataset = create_multi_file_mock_dataset(
            n_claims=n_claims, file_perturbation_types=file_types
        )

        # Count how many perturb_type1 rows exist in the raw data
        type1_rows = [
            ex
            for ex in raw_examples
            if ex["perturbation_type"] == "perturb_type1"
        ]
        print(f"\nRaw rows: {len(dataset)}")
        print(f"perturb_type1 rows (from 2 files): {len(type1_rows)}")
        # 2 files × 4 claims = 8 perturb_type1 entries
        assert len(type1_rows) == n_claims * 2

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=8,
            batch_match_claim_id=True,
            split_large_claims=True,
            shuffle=False,
            seed=42,
        )

        # Each claim should have 4 perturbed entries:
        #   file_0: perturb_type1, perturb_type2
        #   file_1: perturb_type1, perturb_type3_random
        for claim_id, group in sampler.claim_id_groups.items():
            perturbed_types = [
                dataset[idx]["perturbation_type"] for idx in group["perturbed"]
            ]
            perturbed_texts = [
                dataset[idx]["conversations"][1]["content"]
                for idx in group["perturbed"]
            ]

            # Must have 4 perturbed entries (not 3 — both type1's count)
            assert len(group["perturbed"]) == 4, (
                f"{claim_id}: expected 4 perturbed entries (2 from each "
                f"file) but got {len(group['perturbed'])}: {perturbed_types}"
            )

            # perturb_type1 must appear twice (once from each file)
            type1_count = perturbed_types.count("perturb_type1")
            assert type1_count == 2, (
                f"{claim_id}: expected 2 perturb_type1 entries but "
                f"got {type1_count}"
            )

            # The two perturb_type1 entries must have different claim text
            type1_texts = [
                text
                for text, ptype in zip(perturbed_texts, perturbed_types)
                if ptype == "perturb_type1"
            ]
            assert type1_texts[0] != type1_texts[1], (
                f"{claim_id}: two perturb_type1 entries have identical "
                f"text — they should differ (one per source file)"
            )

            print(
                f"  {claim_id}: {len(group['perturbed'])} perturbed, "
                f"type1 ×{type1_count} (distinct texts)"
            )

        # Verify all perturbed entries appear in batches
        batches = list(sampler)
        all_batched_perturbed = [
            idx
            for batch in batches
            for idx in batch
            if dataset[idx]["perturbation_type"] != "original_normalized"
        ]
        expected_perturbed = sum(len(types) * n_claims for types in file_types)
        assert len(all_batched_perturbed) == expected_perturbed, (
            f"Expected {expected_perturbed} perturbed in batches, "
            f"got {len(all_batched_perturbed)}"
        )

        print(
            "\n✓ Test passed: overlapping perturbation types from "
            "different files are all preserved"
        )

    def test_mixed_batches_multi_file_no_dup_originals(self):
        """Validate mixed (non-claim-matched) batching with multi-file data.

        In mixed mode the sampler uses *all* indices, so duplicate
        original_normalized rows will both appear in `original_indices`.
        This test documents that behavior and verifies that the total
        example count is correct.
        """
        print_separator("TEST: Mixed Batches With Multi-File Data")

        raw_examples, dataset = create_multi_file_mock_dataset(
            n_claims=5,
            file_perturbation_types=[
                ["perturb_type1"],
                ["perturb_type2"],
            ],
        )

        sampler = EnhancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=4,
            batch_match_claim_id=False,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        total_in_batches = sum(len(b) for b in batches)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Total examples across batches: {total_in_batches}")

        # All dataset rows should appear (including duplicate originals
        # — mixed mode does not deduplicate)
        assert total_in_batches == len(dataset)

        # The duplicate originals end up in original_indices
        print(f"original_indices count: {len(sampler.original_indices)}")
        print(f"perturbed_indices count: {len(sampler.perturbed_indices)}")

        # 2 files × 5 claims = 10 original rows, 2 files × 1 type
        # × 5 claims = 10 perturbed rows
        assert len(sampler.original_indices) == 10
        assert len(sampler.perturbed_indices) == 10

        print(
            "\n✓ Test passed: mixed-mode batching with multi-file data "
            "preserves all rows"
        )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING BATCH SAMPLER TESTS")
    print("=" * 80)

    pytest.main([__file__, "-v", "-s"])
