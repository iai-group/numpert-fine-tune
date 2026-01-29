import pytest
from datasets import Dataset
from collections import Counter
from code.llm_finetune.class_balancer import DatasetBalancer


def print_separator(title: str):
    """Print a nice separator for test output."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_mock_dataset(
    n_original_true: int = 100,
    n_original_false: int = 100,
    n_t_to_t_per_original: int = 2,
    n_t_to_f_per_original: int = 2,
    n_f_to_f_true_per_original: int = 2,
    n_f_to_f_false_per_original: int = 1,
) -> Dataset:
    """Create a mock dataset with controlled perturbations.

    Args:
        n_original_true: Number of original True examples
        n_original_false: Number of original False examples
        n_t_to_t_per_original: Number of T→T perturbations per True original
        n_t_to_f_per_original: Number of T→F perturbations per True original
        n_f_to_f_true_per_original: Number of F→F (*_true) perturbations per False original
        n_f_to_f_false_per_original: Number of F→F (*_false) perturbations per False original

    Returns:
        Mock dataset
    """
    examples = []

    # Create original True examples with perturbations
    for i in range(n_original_true):
        claim_id = f"true_{i:05d}"

        # Original
        examples.append(
            {
                "claim_id": claim_id,
                "perturbation_type": "original",
                "conversations": [
                    {"role": "system", "content": "You are a fact checker."},
                    {"role": "user", "content": f"Claim {i}"},
                    {"role": "assistant", "content": "True"},
                ],
            }
        )

        # T→T perturbations (use *_true perturbation types)
        for j in range(n_t_to_t_per_original):
            perturb_types = [
                "numeration_true",
                "approximation_true",
                "range_true",
            ]
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": perturb_types[j % len(perturb_types)],
                    "conversations": [
                        {
                            "role": "system",
                            "content": "You are a fact checker.",
                        },
                        {
                            "role": "user",
                            "content": f"Claim {i} perturbed T→T {j}",
                        },
                        {"role": "assistant", "content": "True"},
                    ],
                }
            )

        # T→F perturbations (use *_false perturbation types)
        for j in range(n_t_to_f_per_original):
            perturb_types = [
                "numeration_false",
                "approximation_false",
                "range_false",
                "masking_false",
                "random_replacement_false",
            ]
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": perturb_types[j % len(perturb_types)],
                    "conversations": [
                        {
                            "role": "system",
                            "content": "You are a fact checker.",
                        },
                        {
                            "role": "user",
                            "content": f"Claim {i} perturbed T→F {j}",
                        },
                        {"role": "assistant", "content": "False"},
                    ],
                }
            )

    # Create original False examples with perturbations
    for i in range(n_original_false):
        claim_id = f"false_{i:05d}"

        # Original
        examples.append(
            {
                "claim_id": claim_id,
                "perturbation_type": "original",
                "conversations": [
                    {"role": "system", "content": "You are a fact checker."},
                    {"role": "user", "content": f"False claim {i}"},
                    {"role": "assistant", "content": "False"},
                ],
            }
        )

        # F→F (*_true) perturbations - preferred, more similar to original
        for j in range(n_f_to_f_true_per_original):
            perturb_types = [
                "numeration_true",
                "approximation_true",
                "range_true",
            ]
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": perturb_types[j % len(perturb_types)],
                    "conversations": [
                        {
                            "role": "system",
                            "content": "You are a fact checker.",
                        },
                        {
                            "role": "user",
                            "content": f"False claim {i} perturbed F→F true {j}",
                        },
                        {"role": "assistant", "content": "False"},
                    ],
                }
            )

        # F→F (*_false) perturbations - exaggerated, should be ignored by balancer
        for j in range(n_f_to_f_false_per_original):
            perturb_types = [
                "numeration_false",
                "approximation_false",
                "range_false",
            ]
            examples.append(
                {
                    "claim_id": claim_id,
                    "perturbation_type": perturb_types[j % len(perturb_types)],
                    "conversations": [
                        {
                            "role": "system",
                            "content": "You are a fact checker.",
                        },
                        {
                            "role": "user",
                            "content": f"False claim {i} perturbed F→F false {j}",
                        },
                        {"role": "assistant", "content": "False"},
                    ],
                }
            )

    return Dataset.from_list(examples)


def analyze_dataset(dataset: Dataset) -> dict:
    """Analyze dataset distribution."""
    stats = {
        "total": len(dataset),
        "original_true": 0,
        "original_false": 0,
        "t_to_t": 0,
        "t_to_f": 0,
        "f_to_f_true": 0,
        "f_to_f_false": 0,
        "true_total": 0,
        "false_total": 0,
    }

    for example in dataset:
        perturb_type = example["perturbation_type"]
        label = example["conversations"][-1]["content"]

        # Count by label
        if label == "True":
            stats["true_total"] += 1
        else:
            stats["false_total"] += 1

        # Count by category
        if perturb_type == "original":
            if label == "True":
                stats["original_true"] += 1
            else:
                stats["original_false"] += 1
        else:
            # Determine original label from claim_id
            claim_id = example["claim_id"]
            original_was_true = claim_id.startswith("true_")

            if original_was_true and label == "True":
                stats["t_to_t"] += 1
            elif original_was_true and label == "False":
                stats["t_to_f"] += 1
            elif not original_was_true and label == "False":
                # Check if it's *_true or *_false perturbation
                if perturb_type.endswith("_true"):
                    stats["f_to_f_true"] += 1
                else:
                    stats["f_to_f_false"] += 1

    return stats


def print_stats(stats: dict, title: str = "Dataset Statistics"):
    """Pretty print dataset statistics."""
    print(f"\n{title}:")
    print(f"  Total examples: {stats['total']}")
    print(f"\n  Original examples:")
    print(f"    True:  {stats['original_true']}")
    print(f"    False: {stats['original_false']}")
    print(f"\n  Perturbed examples:")
    print(f"    T→T: {stats['t_to_t']}")
    print(f"    T→F: {stats['t_to_f']}")
    print(f"    F→F (*_true):  {stats['f_to_f_true']}")
    print(f"    F→F (*_false): {stats['f_to_f_false']}")
    print(f"\n  Label distribution:")
    print(
        f"    True:  {stats['true_total']} ({100*stats['true_total']/stats['total']:.1f}%)"
    )
    print(
        f"    False: {stats['false_total']} ({100*stats['false_total']/stats['total']:.1f}%)"
    )


class TestDatasetBalancer:
    """Test suite for DatasetBalancer."""

    def test_no_balancing_default_ratio(self):
        """Test with no label balancing and default 1:1 perturb ratio."""
        print_separator("TEST: No Balancing, Default 1:1 Ratio")

        # Create dataset: 100 True, 100 False, each with 2 T→T, 2 T→F, 2 F→F_true
        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=100,
            n_t_to_t_per_original=2,
            n_t_to_f_per_original=2,
            n_f_to_f_true_per_original=2,
            n_f_to_f_false_per_original=1,  # Should be ignored
        )

        print(f"\nInput dataset size: {len(dataset)}")
        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset")

        # Balance with default settings
        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=0.67,
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset")

        # Verify: perturb_ratio=1.0 means 1:1 ratio of perturbed to non-perturbed
        # 100 original True → 100 perturbed True (all T→T)
        # 100 original False → 100 perturbed False (67 T→F, 33 F→F)

        assert output_stats["original_true"] == 100
        assert output_stats["original_false"] == 100
        assert output_stats["t_to_t"] == 100

        # T→F should be 67% of 100 = 67
        expected_t_to_f = int(100 * 0.67)
        assert output_stats["t_to_f"] == expected_t_to_f

        # F→F should be 33% of 100 = 33
        expected_f_to_f = 100 - expected_t_to_f
        assert output_stats["f_to_f_true"] == expected_f_to_f

        # F→F_false should not be included
        assert output_stats["f_to_f_false"] == 0

        # Total True: 100 original + 100 T→T = 200
        # Total False: 100 original + 67 T→F + 33 F→F = 200
        assert output_stats["true_total"] == 200
        assert output_stats["false_total"] == 200

        print("\n✓ Test passed: Correct 1:1 ratio with proper T→F/F→F split")

    def test_balanced_labels(self):
        """Test label balancing with unequal original distribution."""
        print_separator("TEST: Balanced Labels (Unequal Originals)")

        # Create imbalanced dataset: 200 True, 50 False
        dataset = create_mock_dataset(
            n_original_true=200,
            n_original_false=50,
            n_t_to_t_per_original=2,
            n_t_to_f_per_original=2,
            n_f_to_f_true_per_original=2,
        )

        print(f"\nInput dataset size: {len(dataset)}")
        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset (Imbalanced)")

        # Balance labels
        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=True,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=0.67,
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset (Balanced)")

        # After perturb_ratio:
        # True: 200 original + 200 T→T = 400
        # False: 50 original + 33 T→F + 17 F→F = 100

        # After balancing (downsample True to 100):
        # Downsample ratio = 100/400 = 0.25
        # True: 50 original + 50 T→T = 100
        # False: 50 original + 33 T→F + 17 F→F = 100

        expected_original_true = int(200 * 0.25)
        expected_t_to_t = int(200 * 0.25)

        assert output_stats["original_true"] == expected_original_true
        assert output_stats["t_to_t"] == expected_t_to_t
        assert output_stats["original_false"] == 50

        # Total should be balanced
        assert output_stats["true_total"] == output_stats["false_total"]

        print(
            f"\n✓ Test passed: Labels balanced (True={output_stats['true_total']}, "
            f"False={output_stats['false_total']})"
        )

    def test_custom_perturb_ratio(self):
        """Test with custom perturbation ratio (2:1)"""
        print_separator("TEST: Custom Perturb Ratio (2:1)")

        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=100,
            n_t_to_t_per_original=5,  # Plenty available
            n_t_to_f_per_original=5,
            n_f_to_f_true_per_original=5,
        )

        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset")

        # Use 2:1 perturb ratio
        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=2.0,  # 2 perturbed for each original
            false_perturb_t_to_f_ratio=0.5,  # 50/50 split
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset")

        # Expected:
        # 100 original True → 200 T→T
        # 100 original False → 200 perturbed False (100 T→F, 100 F→F)

        assert output_stats["original_true"] == 100
        assert output_stats["original_false"] == 100
        assert output_stats["t_to_t"] == 200

        expected_t_to_f = int(200 * 0.5)
        expected_f_to_f = 200 - expected_t_to_f

        assert output_stats["t_to_f"] == expected_t_to_f
        assert output_stats["f_to_f_true"] == expected_f_to_f

        # Total: 100 original True + 200 T→T = 300 True
        #        100 original False + 100 T→F + 100 F→F = 300 False
        assert output_stats["true_total"] == 300
        assert output_stats["false_total"] == 300

        print(f"\n✓ Test passed: 2:1 perturb ratio with 50/50 T→F/F→F split")

    def test_custom_t_to_f_ratio(self):
        """Test with different T→F ratios."""
        print_separator("TEST: Custom T→F Ratios")

        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=100,
            n_t_to_t_per_original=3,
            n_t_to_f_per_original=3,
            n_f_to_f_true_per_original=3,
        )

        # Test different T→F ratios
        ratios_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]

        for t_to_f_ratio in ratios_to_test:
            print(f"\n--- Testing T→F ratio: {t_to_f_ratio} ---")

            balancer = DatasetBalancer(
                dataset=dataset,
                balance_labels=False,
                perturb_ratio=1.0,
                false_perturb_t_to_f_ratio=t_to_f_ratio,
                seed=42,
            )

            balanced = balancer.balance()
            output_stats = analyze_dataset(balanced)

            # 100 original False → 100 perturbed False
            total_false_perturbed = 100
            expected_t_to_f = int(total_false_perturbed * t_to_f_ratio)
            expected_f_to_f = total_false_perturbed - expected_t_to_f

            print(
                f"  Expected T→F: {expected_t_to_f}, Actual: {output_stats['t_to_f']}"
            )
            print(
                f"  Expected F→F: {expected_f_to_f}, Actual: {output_stats['f_to_f_true']}"
            )

            assert output_stats["t_to_f"] == expected_t_to_f
            assert output_stats["f_to_f_true"] == expected_f_to_f

        print("\n✓ Test passed: All T→F ratios work correctly")

    def test_insufficient_perturbations(self):
        """Test when there aren't enough perturbations available."""
        print_separator("TEST: Insufficient Perturbations")

        # Create dataset with limited perturbations
        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=100,
            n_t_to_t_per_original=0,  # No T→T available!
            n_t_to_f_per_original=1,  # Limited T→F
            n_f_to_f_true_per_original=0,  # No F→F available!
        )

        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset (Limited Perturbations)")

        # Request 1:1 ratio (will use all available)
        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=0.67,
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset (Used All Available)")

        # Should use all available
        assert output_stats["t_to_t"] == 0  # None available
        assert (
            output_stats["t_to_f"] == 67
        )  # Requested 67, but only 100 available
        assert output_stats["f_to_f_true"] == 0  # None available

        print("\n✓ Test passed: Handles insufficient perturbations gracefully")

    def test_f_to_f_only_uses_true_perturbations(self):
        """Test that F→F only uses *_true perturbations, ignores *_false."""
        print_separator("TEST: F→F Uses Only *_true Perturbations")

        dataset = create_mock_dataset(
            n_original_true=50,
            n_original_false=50,
            n_t_to_t_per_original=1,
            n_t_to_f_per_original=1,
            n_f_to_f_true_per_original=3,  # 3 *_true perturbations
            n_f_to_f_false_per_original=5,  # 5 *_false perturbations (should be ignored)
        )

        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset")

        print(
            f"\nNote: Input has {input_stats['f_to_f_false']} F→F (*_false) perturbations"
        )
        print(f"      These should NOT be included in the output")

        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=0.5,  # 50% T→F, 50% F→F
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset")

        # 50 original False → 50 perturbed (25 T→F, 25 F→F)
        # F→F should ONLY come from *_true perturbations

        assert (
            output_stats["f_to_f_false"] == 0
        ), "F→F (*_false) should not be included!"
        assert (
            output_stats["f_to_f_true"] == 25
        ), "F→F should only use *_true perturbations"

        print("\n✓ Test passed: F→F correctly uses only *_true perturbations")

    def test_edge_case_zero_perturb_ratio(self):
        """Test with perturb_ratio = 0 (only originals)"""
        print_separator("TEST: Edge Case - Zero Perturb Ratio")

        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=100,
            n_t_to_t_per_original=5,
            n_t_to_f_per_original=5,
            n_f_to_f_true_per_original=5,
        )

        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=0.0,  # No perturbations!
            false_perturb_t_to_f_ratio=0.67,
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset (Originals Only)")

        # Should only have originals
        assert output_stats["original_true"] == 100
        assert output_stats["original_false"] == 100
        assert output_stats["t_to_t"] == 0
        assert output_stats["t_to_f"] == 0
        assert output_stats["f_to_f_true"] == 0
        assert output_stats["total"] == 200

        print("\n✓ Test passed: Zero perturb_ratio gives only originals")

    def test_complex_scenario(self):
        """Test complex scenario: imbalanced + custom ratios + balancing"""
        print_separator("TEST: Complex Scenario")

        # 150 True originals, 80 False originals
        dataset = create_mock_dataset(
            n_original_true=150,
            n_original_false=80,
            n_t_to_t_per_original=4,
            n_t_to_f_per_original=4,
            n_f_to_f_true_per_original=4,
        )

        input_stats = analyze_dataset(dataset)
        print_stats(input_stats, "Input Dataset (Complex)")

        # Balance labels with 1.5:1 perturb ratio and 0.4 T→F ratio
        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=True,
            perturb_ratio=1.5,
            false_perturb_t_to_f_ratio=0.4,
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset (Complex Balanced)")

        # Step 1: Apply perturb_ratio=1.5
        # True: 150 original + 225 T→T = 375
        # False: 80 original + 48 T→F + 72 F→F = 200

        # Step 2: Balance (downsample True to 200)
        # Downsample ratio = 200/375 = 0.533...
        # True: 80 original + 120 T→T = 200
        # False: 80 original + 48 T→F + 72 F→F = 200

        expected_original_true = int(150 * (200 / 375))
        expected_t_to_t = int(225 * (200 / 375))

        # Allow small rounding error
        assert abs(output_stats["original_true"] - expected_original_true) <= 1
        assert abs(output_stats["t_to_t"] - expected_t_to_t) <= 1

        # Labels should be balanced
        assert (
            abs(output_stats["true_total"] - output_stats["false_total"]) <= 2
        )

        print(f"\n✓ Test passed: Complex scenario handled correctly")
        print(
            f"   Final balance: True={output_stats['true_total']}, "
            f"False={output_stats['false_total']}"
        )


class TestDatasetBalancerEdgeCases:
    """Additional edge case tests."""

    def test_empty_dataset(self):
        """Test with empty dataset."""
        print_separator("TEST: Empty Dataset")

        dataset = Dataset.from_list([])

        balancer = DatasetBalancer(
            dataset=dataset, balance_labels=False, perturb_ratio=1.0, seed=42
        )

        balanced = balancer.balance()

        assert len(balanced) == 0
        print("\n✓ Test passed: Empty dataset handled")

    def test_only_originals_no_perturbations(self):
        """Test dataset with only original examples."""
        print_separator("TEST: Only Originals, No Perturbations")

        dataset = create_mock_dataset(
            n_original_true=50,
            n_original_false=50,
            n_t_to_t_per_original=0,
            n_t_to_f_per_original=0,
            n_f_to_f_true_per_original=0,
        )

        balancer = DatasetBalancer(
            dataset=dataset,
            balance_labels=False,
            perturb_ratio=2.0,  # Request 2:1 but none available
            seed=42,
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset")

        # Should have originals but no perturbations
        assert output_stats["original_true"] == 50
        assert output_stats["original_false"] == 50
        assert output_stats["total"] == 100

        print("\n✓ Test passed: Handles dataset with no perturbations")

    def test_all_true_labels(self):
        """Test dataset with only True labels."""
        print_separator("TEST: All True Labels")

        dataset = create_mock_dataset(
            n_original_true=100,
            n_original_false=0,  # No False examples
            n_t_to_t_per_original=2,
            n_t_to_f_per_original=2,
        )

        balancer = DatasetBalancer(
            dataset=dataset, balance_labels=True, perturb_ratio=1.0, seed=42
        )

        balanced = balancer.balance()
        output_stats = analyze_dataset(balanced)
        print_stats(output_stats, "Output Dataset")

        # With balance_labels=True and no False examples, should downsample to 0
        assert output_stats["false_total"] == 0

        print("\n✓ Test passed: Handles all-True dataset")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING DATASET BALANCER TESTS")
    print("=" * 80)

    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
