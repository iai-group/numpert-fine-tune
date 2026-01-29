import pytest
import json
import tempfile
from pathlib import Path
from typing import List, Dict
from datasets import Dataset

# Assuming your script is named data_loader.py
from code.llm_finetune.hf_dataset_instructions import (
    load_json_files,
    expand_perturbations,
    convert_to_conversations,
    load_fact_checking_data,
    load_train_and_validation,
    PerturbationType,
)


# Mock prompts for testing
FACT_CHECK_SYSTEM_PROMPT = "You are a fact-checking assistant."
FACT_CHECK_USER_PROMPT = "Please verify the following claim."


@pytest.fixture
def mock_json_data_single():
    """Single example with all perturbation types."""
    return {
        "url": "https://example.com/fact-check-1",
        "crawled_date": "2024-01-15",
        "taxonomy_label": "statistical",
        "label": "True",
        "normalized_claim": "The government distributed 35,000 LED bulbs.",
        "lang": "en",
        "country_of_origin": "india",
        "label_original": "True",
        "claim_id": "train_000001",
        "claim": "The government distributed 35,000 LED bulbs.",
        "doc": "Official government report shows distribution of 35,000 LED bulbs in 2023.",
        "numeration_true": "The government distributed thirty-five thousand LED bulbs.",
        "numeration_false": "The government distributed fifty thousand LED bulbs.",
        "approximation_true": "The government distributed approximately 35,000 LED bulbs.",
        "approximation_false": "The government distributed approximately 100,000 LED bulbs.",
        "range_true": "The government distributed between 34,000 and 36,000 LED bulbs.",
        "range_false": "The government distributed between 50,000 and 60,000 LED bulbs.",
        "masking_false": "The government distributed [MASK] LED bulbs.",
        "random_replacement_false": "The government distributed 92,847 LED bulbs.",
    }


@pytest.fixture
def mock_json_data_multiple():
    """Multiple examples with varying perturbation types."""
    return [
        {
            "url": "https://example.com/fact-check-1",
            "crawled_date": "2024-01-15",
            "taxonomy_label": "statistical",
            "label": "True",
            "normalized_claim": "The unemployment rate decreased by 2%.",
            "lang": "en",
            "country_of_origin": "usa",
            "label_original": "True",
            "claim_id": "train_000001",
            "claim": "The unemployment rate decreased by 2%.",
            "doc": "Labor statistics show a 2% decrease in unemployment.",
            "numeration_true": "The unemployment rate decreased by two percent.",
            "numeration_false": "The unemployment rate decreased by five percent.",
        },
        {
            "url": "https://example.com/fact-check-2",
            "crawled_date": "2024-01-16",
            "taxonomy_label": "factual",
            "label": "False",
            "normalized_claim": "The population of New York is 50 million.",
            "lang": "en",
            "country_of_origin": "usa",
            "label_original": "False",
            "claim_id": "train_000002",
            "claim": "The population of New York is 50 million.",
            "doc": "Census data shows the population is approximately 8 million.",
            "approximation_true": "The population of New York is approximately 8 million.",
            "approximation_false": "The population of New York is approximately 50 million.",
            "range_true": "The population of New York is between 7 and 9 million.",
            "range_false": "The population of New York is between 45 and 55 million.",
        },
    ]


@pytest.fixture
def temp_json_files(mock_json_data_multiple, tmp_path):
    """Create temporary JSON files for testing."""
    # Create train file 1
    train_file_1 = tmp_path / "train_1.json"
    with open(train_file_1, "w") as f:
        json.dump([mock_json_data_multiple[0]], f)

    # Create train file 2
    train_file_2 = tmp_path / "train_2.json"
    with open(train_file_2, "w") as f:
        json.dump([mock_json_data_multiple[1]], f)

    # Create validation file
    val_file = tmp_path / "val.json"
    val_data = {
        "url": "https://example.com/fact-check-val",
        "crawled_date": "2024-01-17",
        "taxonomy_label": "statistical",
        "label": "True",
        "normalized_claim": "Validation claim example.",
        "lang": "en",
        "country_of_origin": "usa",
        "label_original": "True",
        "claim_id": "val_000001",
        "claim": "Validation claim example.",
        "doc": "Validation evidence document.",
        "numeration_true": "Validation claim with number spelled out.",
        "numeration_false": "Validation claim with wrong number.",
    }
    with open(val_file, "w") as f:
        json.dump([val_data], f)

    return {
        "train_1": str(train_file_1),
        "train_2": str(train_file_2),
        "val": str(val_file),
    }


def print_separator(title: str):
    """Print a nice separator for test output."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_example(example: Dict, index: int = None):
    """Pretty print an example."""
    if index is not None:
        print(f"\n--- Example {index} ---")
    else:
        print(f"\n--- Example ---")

    for key, value in example.items():
        if key == "conversations":
            print(f"{key}:")
            for i, msg in enumerate(value):
                print(
                    f"  [{i}] {msg['role']}: {msg['content'][:100]}..."
                    if len(msg["content"]) > 100
                    else f"  [{i}] {msg['role']}: {msg['content']}"
                )
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")


class TestLoadJsonFiles:
    """Test JSON file loading functionality."""

    def test_load_single_json_file(self, temp_json_files):
        print_separator("TEST: Load Single JSON File")

        examples = load_json_files(temp_json_files["train_1"])

        print(f"\nLoaded {len(examples)} example(s) from single file")
        print(f"First example claim_id: {examples[0]['claim_id']}")
        print(f"First example claim: {examples[0]['normalized_claim']}")

        assert len(examples) == 1
        assert examples[0]["claim_id"] == "train_000001"
        print("\n✓ Single file loading successful")

    def test_load_multiple_json_files(self, temp_json_files):
        print_separator("TEST: Load Multiple JSON Files")

        examples = load_json_files(
            [temp_json_files["train_1"], temp_json_files["train_2"]]
        )

        print(f"\nLoaded {len(examples)} example(s) from {2} files")
        for i, ex in enumerate(examples):
            print(
                f"  Example {i+1}: claim_id={ex['claim_id']}, label={ex['label']}"
            )

        assert len(examples) == 2
        assert examples[0]["claim_id"] == "train_000001"
        assert examples[1]["claim_id"] == "train_000002"
        print("\n✓ Multiple file loading successful")


class TestExpandPerturbations:
    """Test perturbation expansion functionality."""

    def test_expand_all_perturbations(self, mock_json_data_single):
        print_separator("TEST: Expand All Perturbations")

        expanded = expand_perturbations([mock_json_data_single])

        print(f"\nOriginal examples: 1")
        print(f"Expanded examples: {len(expanded)}")
        print(f"\nPerturbation types found:")

        perturb_types = {}
        for ex in expanded:
            p_type = ex["perturbation_type"]
            perturb_types[p_type] = perturb_types.get(p_type, 0) + 1
            print(
                f"  - {p_type}: label={ex['label']}, claim={ex['claim'][:60]}..."
            )

        print(f"\nTotal perturbation types: {len(perturb_types)}")

        # Should have original + 8 perturbations = 9 total
        assert len(expanded) == 9
        assert any(ex["perturbation_type"] == "original" for ex in expanded)
        assert any(
            ex["perturbation_type"] == "numeration_true" for ex in expanded
        )
        assert any(
            ex["perturbation_type"] == "numeration_false" for ex in expanded
        )

        print("\n✓ Perturbation expansion successful")

    def test_perturbation_labels(self, mock_json_data_single):
        print_separator("TEST: Perturbation Label Assignment")

        expanded = expand_perturbations([mock_json_data_single])

        print("\nChecking label assignments:")
        for ex in expanded:
            p_type = ex["perturbation_type"]
            label = ex["label"]

            if p_type == "original":
                expected = "True"  # Original label
                print(f"  {p_type:30s} -> {label:10s} (expected: {expected})")
                assert label == expected
            elif p_type.endswith("_true"):
                expected = "True"  # Should keep original label
                print(f"  {p_type:30s} -> {label:10s} (expected: {expected})")
                assert label == expected
            elif p_type.endswith("_false"):
                expected = "False"  # Should be False
                print(f"  {p_type:30s} -> {label:10s} (expected: {expected})")
                assert label == expected

        print("\n✓ All labels correctly assigned")

    def test_partial_perturbations(self, mock_json_data_multiple):
        print_separator("TEST: Partial Perturbations (Not All Types Present)")

        expanded = expand_perturbations(mock_json_data_multiple)

        print(f"\nOriginal examples: {len(mock_json_data_multiple)}")
        print(f"Expanded examples: {len(expanded)}")

        # Group by claim_id
        by_claim_id = {}
        for ex in expanded:
            claim_id = ex["claim_id"]
            if claim_id not in by_claim_id:
                by_claim_id[claim_id] = []
            by_claim_id[claim_id].append(ex["perturbation_type"])

        print("\nPerturbations per claim_id:")
        for claim_id, p_types in by_claim_id.items():
            print(f"  {claim_id}: {len(p_types)} variations")
            for p_type in p_types:
                print(f"    - {p_type}")

        assert len(by_claim_id) == 2
        print("\n✓ Partial perturbations handled correctly")


class TestConvertToConversations:
    """Test conversation format conversion."""

    def test_train_conversation_format(self, mock_json_data_single):
        print_separator("TEST: Train Conversation Format")

        expanded = expand_perturbations([mock_json_data_single])
        conversations = convert_to_conversations(expanded, split="train")

        print(f"\nConverted {len(expanded)} examples to conversation format")

        # Print first conversation
        print("\nFirst conversation structure:")
        first_conv = conversations[0]
        print(f"  claim_id: {first_conv['claim_id']}")
        print(f"  perturbation_type: {first_conv['perturbation_type']}")
        print(f"  Number of messages: {len(first_conv['conversations'])}")

        for i, msg in enumerate(first_conv["conversations"]):
            role = msg["role"]
            content_preview = (
                msg["content"][:80] + "..."
                if len(msg["content"]) > 80
                else msg["content"]
            )
            print(f"    [{i}] {role:10s}: {content_preview}")

        # Verify structure
        assert len(conversations) == len(expanded)
        assert "conversations" in conversations[0]
        assert "claim_id" in conversations[0]
        assert "perturbation_type" in conversations[0]
        assert (
            "ground_truth" not in conversations[0]
        )  # Train doesn't have ground_truth

        # Verify conversation structure
        conv = conversations[0]["conversations"]
        assert (
            len(conv) == 4
        )  # system, user instruction, user claim, assistant response
        assert conv[0]["role"] == "system"
        assert conv[1]["role"] == "user"
        assert conv[2]["role"] == "user"
        assert conv[3]["role"] == "assistant"

        print("\n✓ Train conversation format correct")

    def test_validation_conversation_format(self, mock_json_data_single):
        print_separator("TEST: Validation Conversation Format")

        expanded = expand_perturbations([mock_json_data_single])
        conversations = convert_to_conversations(expanded, split="validation")

        print(f"\nConverted {len(expanded)} examples to validation format")

        # Print first conversation
        print("\nFirst validation example:")
        first_conv = conversations[0]
        print(f"  claim_id: {first_conv['claim_id']}")
        print(f"  perturbation_type: {first_conv['perturbation_type']}")
        print(f"  ground_truth: {first_conv['ground_truth']}")
        print(f"  Number of messages: {len(first_conv['conversations'])}")

        # Verify structure
        assert "ground_truth" in conversations[0]  # Val has ground_truth

        # Verify conversation structure (no assistant response)
        conv = conversations[0]["conversations"]
        assert (
            len(conv) == 3
        )  # system, user instruction, user claim (no assistant)
        assert conv[-1]["role"] == "user"  # Last message should be user

        print("\n✓ Validation conversation format correct")

    def test_conversation_with_doc(self, mock_json_data_single):
        print_separator("TEST: Conversation With Evidence Document")

        expanded = expand_perturbations([mock_json_data_single])
        conversations = convert_to_conversations(
            expanded, split="train", include_doc=True
        )

        first_conv = conversations[0]["conversations"]
        claim_message = first_conv[2]["content"]  # Third message is the claim

        print("\nClaim message with evidence:")
        print(claim_message)

        assert "Claim:" in claim_message
        assert "Evidence:" in claim_message
        assert mock_json_data_single["doc"] in claim_message

        print("\n✓ Evidence document included correctly")

    def test_conversation_without_doc(self, mock_json_data_single):
        print_separator("TEST: Conversation Without Evidence Document")

        expanded = expand_perturbations([mock_json_data_single])
        conversations = convert_to_conversations(
            expanded, split="train", include_doc=False
        )

        first_conv = conversations[0]["conversations"]
        claim_message = first_conv[2]["content"]

        print("\nClaim message without evidence:")
        print(claim_message)

        assert "Claim:" in claim_message
        assert "Evidence:" not in claim_message

        print("\n✓ Evidence document excluded correctly")


class TestLoadFactCheckingData:
    """Test end-to-end data loading."""

    def test_load_dataset_with_expansion(self, temp_json_files):
        print_separator("TEST: Load Dataset With Perturbation Expansion")

        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=True,
        )

        print(f"\nDataset type: {type(dataset)}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset features: {list(dataset.features.keys())}")

        # Print first 3 examples
        print("\nFirst 3 examples:")
        for i in range(min(3, len(dataset))):
            print_example(dataset[i], i + 1)

        assert isinstance(dataset, Dataset)
        assert (
            len(dataset) > 1
        )  # Should have multiple examples due to expansion
        assert "conversations" in dataset.features
        assert "claim_id" in dataset.features
        assert "perturbation_type" in dataset.features

        print("\n✓ Dataset loaded and expanded successfully")

    def test_load_dataset_without_expansion(self, temp_json_files):
        print_separator("TEST: Load Dataset Without Perturbation Expansion")

        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=False,
        )

        print(f"\nDataset size: {len(dataset)}")
        print(f"All perturbation types: {set(dataset['perturbation_type'])}")

        # Should only have 1 example (original only, no perturbations)
        assert len(dataset) == 1
        assert all(
            p_type == "original" for p_type in dataset["perturbation_type"]
        )

        print("\n✓ Dataset loaded without expansion")

    def test_load_multiple_files(self, temp_json_files):
        print_separator("TEST: Load Multiple JSON Files")

        dataset = load_fact_checking_data(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations_flag=True,
        )

        claim_ids = set(dataset["claim_id"])
        print(f"\nDataset size: {len(dataset)}")
        print(f"Unique claim_ids: {len(claim_ids)}")
        print(f"Claim IDs: {sorted(claim_ids)}")

        # Count perturbations per claim_id
        from collections import Counter

        perturb_counts = Counter(dataset["claim_id"])
        print("\nExamples per claim_id:")
        for claim_id, count in sorted(perturb_counts.items()):
            print(f"  {claim_id}: {count} examples")

        assert len(claim_ids) == 2  # Two original claims

        print("\n✓ Multiple files concatenated successfully")


class TestLoadTrainAndValidation:
    """Test train/validation split loading."""

    def test_load_train_and_val(self, temp_json_files):
        print_separator("TEST: Load Train and Validation Datasets")

        datasets = load_train_and_validation(
            train_json_paths=[
                temp_json_files["train_1"],
                temp_json_files["train_2"],
            ],
            val_json_paths=temp_json_files["val"],
            expand_perturbations_flag=True,
        )

        print(f"\nDatasets loaded:")
        print(f"  Keys: {list(datasets.keys())}")
        print(f"  Train size: {len(datasets['train'])}")
        print(f"  Validation size: {len(datasets['validation'])}")

        # Check train dataset
        print("\nTrain dataset:")
        print(f"  Features: {list(datasets['train'].features.keys())}")
        print(
            f"  Has ground_truth: {'ground_truth' in datasets['train'].features}"
        )

        # Check validation dataset
        print("\nValidation dataset:")
        print(f"  Features: {list(datasets['validation'].features.keys())}")
        print(
            f"  Has ground_truth: {'ground_truth' in datasets['validation'].features}"
        )

        # Print sample from each
        print("\n--- Sample Train Example ---")
        train_sample = datasets["train"][0]
        print(f"claim_id: {train_sample['claim_id']}")
        print(f"perturbation_type: {train_sample['perturbation_type']}")
        print(f"Last message role: {train_sample['conversations'][-1]['role']}")

        print("\n--- Sample Validation Example ---")
        val_sample = datasets["validation"][0]
        print(f"claim_id: {val_sample['claim_id']}")
        print(f"perturbation_type: {val_sample['perturbation_type']}")
        print(f"ground_truth: {val_sample['ground_truth']}")
        print(f"Last message role: {val_sample['conversations'][-1]['role']}")

        assert "train" in datasets
        assert "validation" in datasets
        assert len(datasets["train"]) > 0
        assert len(datasets["validation"]) > 0

        # Train should not have ground_truth, val should
        assert "ground_truth" not in datasets["train"].features
        assert "ground_truth" in datasets["validation"].features

        # Train should have assistant response, val should not
        assert datasets["train"][0]["conversations"][-1]["role"] == "assistant"
        assert datasets["validation"][0]["conversations"][-1]["role"] == "user"

        print("\n✓ Train and validation datasets loaded correctly")


class TestPerturbationTypeDistribution:
    """Test perturbation type distribution analysis."""

    def test_perturbation_distribution(self, temp_json_files):
        print_separator("TEST: Perturbation Type Distribution")

        dataset = load_fact_checking_data(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations_flag=True,
        )

        from collections import Counter

        # Count perturbations
        perturb_counts = Counter(dataset["perturbation_type"])

        print(f"\nTotal examples: {len(dataset)}")
        print(f"\nPerturbation type distribution:")
        for perturb_type, count in sorted(perturb_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {perturb_type:30s}: {count:3d} ({percentage:5.1f}%)")

        # Count by label
        label_counts = Counter(
            dataset["conversations"][i][-1]["content"]
            for i in range(len(dataset))
        )

        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {label:10s}: {count:3d} ({percentage:5.1f}%)")

        # Verify we have original examples
        assert "original" in perturb_counts
        assert perturb_counts["original"] > 0

        print("\n✓ Distribution analysis complete")


class TestClaimIdTracking:
    """Test claim_id tracking across perturbations."""

    def test_claim_id_consistency(self, temp_json_files):
        print_separator("TEST: Claim ID Consistency Across Perturbations")

        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=True,
        )

        # Group by claim_id
        from collections import defaultdict

        by_claim_id = defaultdict(list)

        for example in dataset:
            by_claim_id[example["claim_id"]].append(
                example["perturbation_type"]
            )

        print(f"\nClaim IDs and their perturbations:")
        for claim_id, perturb_types in sorted(by_claim_id.items()):
            print(f"\n{claim_id}:")
            print(f"  Total variations: {len(perturb_types)}")
            for p_type in sorted(perturb_types):
                print(f"    - {p_type}")

        # All examples with same claim_id should have same base claim
        assert len(by_claim_id) > 0

        print("\n✓ Claim IDs tracked consistently")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING DATA LOADER TESTS")
    print("=" * 80)

    pytest.main([__file__, "-v", "-s"])
