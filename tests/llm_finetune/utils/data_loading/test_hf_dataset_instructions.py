import pytest
import json
from typing import Dict
from collections import Counter, defaultdict
from datasets import Dataset

from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
    parse_file_config,
)


# ---------------------------------------------------------------------------
# Fixtures — new JSON format matching PERTURBATIONS.md
# ---------------------------------------------------------------------------


@pytest.fixture
def numeration_true_example():
    """Single entry from numeration_True.json (original label = True)."""
    return {
        "crawled_date": "2024-01-15",
        "country_of_origin": "usa",
        "label": True,
        "url": "https://example.com/fact-check-1",
        "lang": "en",
        "claim": "The government distributed 35,000 LED bulbs.",
        "doc": (
            "Official report shows distribution of 35,000 LED bulbs " "in 2023."
        ),
        "taxonomy_label": "statistical",
        "label_original": "True",
        "claim_id": "train_000001",
        "normalized_claim": "The government distributed 35,000 LED bulbs.",
        "perturb_type1_True": (
            "The government distributed thirty-five thousand LED bulbs."
        ),
        "perturb_type2_True": (
            "The government distributed thirty-eight thousand five "
            "hundred LED bulbs."
        ),
    }


@pytest.fixture
def approximation_false_example():
    """Single entry from approximation_False.json (original label = False)."""
    return {
        "crawled_date": "2024-01-16",
        "country_of_origin": "india",
        "label": False,
        "url": "https://example.com/fact-check-2",
        "lang": "en",
        "claim": "Nirmala distributed 35,000 crore LED bulbs.",
        "doc": "Records show only 350 million LED bulbs were distributed.",
        "taxonomy_label": "statistical",
        "label_original": "False",
        "claim_id": "train_000002",
        "normalized_claim": "Nirmala distributed 35,000 crore LED bulbs.",
        "perturb_type1_False": (
            "Nirmala distributed about 35,000 crore LED bulbs."
        ),
        "perturb_type2_False": (
            "Nirmala distributed about 60,000 crore LED bulbs."
        ),
    }


@pytest.fixture
def obfuscation_true_example():
    """Single entry from obfuscation_True.json (original label = True)."""
    return {
        "crawled_date": "2024-01-17",
        "country_of_origin": "usa",
        "label": True,
        "url": "https://example.com/fact-check-3",
        "lang": "en",
        "claim": "Columbus would save $41,000,000 a year.",
        "doc": "Analysis shows Columbus would save $41 million annually.",
        "taxonomy_label": "statistical",
        "label_original": "mostly-true",
        "claim_id": "train_000003",
        "normalized_claim": "Columbus would save $41,000,000 a year.",
        "perturb_type3_random": "Columbus would save $44,715,814 a year.",
        "perturb_type3_masked": "Columbus would save $########## a year.",
        "perturb_type3_removed": "Columbus would save $ a year.",
    }


@pytest.fixture
def negative_true_example():
    """Single entry from negative_True.json (only has type2)."""
    return {
        "crawled_date": "2024-01-18",
        "country_of_origin": "south africa",
        "label": True,
        "url": "https://example.com/fact-check-4",
        "lang": "en",
        "claim": "Close to 40% of South Africans who need jobs are unemployed.",
        "doc": "The expanded unemployment rate stands close to 40%.",
        "taxonomy_label": "statistical",
        "label_original": "correct",
        "claim_id": "train_000004",
        "normalized_claim": (
            "Close to 40% of South Africans who need jobs are " "unemployed."
        ),
        "perturb_type2_True": (
            "Close to -40% of South Africans who need jobs are " "unemployed."
        ),
    }


@pytest.fixture
def temp_json_files(
    numeration_true_example, approximation_false_example, tmp_path
):
    """Create temporary JSON files mimicking the real data split layout."""
    train_file_1 = tmp_path / "numeration_True.json"
    with open(train_file_1, "w") as f:
        json.dump([numeration_true_example], f)

    train_file_2 = tmp_path / "approximation_False.json"
    with open(train_file_2, "w") as f:
        json.dump([approximation_false_example], f)

    val_file = tmp_path / "val_numeration_True.json"
    val_data = {
        "crawled_date": "2024-02-01",
        "country_of_origin": "usa",
        "label": True,
        "url": "https://example.com/val-1",
        "lang": "en",
        "claim": "Validation claim about 100 items.",
        "doc": "Validation evidence about 100 items.",
        "taxonomy_label": "statistical",
        "label_original": "True",
        "claim_id": "val_000001",
        "normalized_claim": "Validation claim about 100 items.",
        "perturb_type1_True": "Validation claim about one hundred items.",
        "perturb_type2_True": (
            "Validation claim about one hundred and ten items."
        ),
    }
    with open(val_file, "w") as f:
        json.dump([val_data], f)

    return {
        "train_1": str(train_file_1),
        "train_2": str(train_file_2),
        "val": str(val_file),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_separator(title: str):
    """Print a nice separator for test output."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_example(example: Dict, index: int = None):  # type: ignore[assignment]
    """Pretty print an example."""
    prefix = (
        f"--- Example {index} ---" if index is not None else "--- Example ---"
    )
    print(f"\n{prefix}")
    for key, value in example.items():
        if key == "conversations":
            print(f"{key}:")
            for i, msg in enumerate(value):
                content = (
                    f"{msg['content'][:100]}..."
                    if len(msg["content"]) > 100
                    else msg["content"]
                )
                print(f"  [{i}] {msg['role']}: {content}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# Tests — FileLoadConfig & parse_file_config
# ---------------------------------------------------------------------------


class TestParseFileConfig:
    """Test CLI-style file config parsing."""

    def test_plain_path(self):
        cfg = parse_file_config("data/train/approx_True.json")
        assert cfg.path == "data/train/approx_True.json"
        assert cfg.perturb_types is None

    def test_path_with_types_shorthand(self):
        cfg = parse_file_config("approx_True.json:type1,type2")
        assert cfg.path == "approx_True.json"
        assert cfg.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_path_with_types_full_name(self):
        cfg = parse_file_config(
            "obf.json:perturb_type3_random,perturb_type3_masked"
        )
        assert cfg.path == "obf.json"
        assert cfg.perturb_types == [
            PerturbationType.PERTURB_TYPE3_RANDOM,
            PerturbationType.PERTURB_TYPE3_MASKED,
        ]

    def test_path_with_empty_types_loads_all(self):
        cfg = parse_file_config("file.json:")
        assert cfg.path == "file.json"
        assert cfg.perturb_types is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation type"):
            parse_file_config("file.json:nonexistent_type")


# ---------------------------------------------------------------------------
# Tests — load_split (end-to-end pipeline)
# ---------------------------------------------------------------------------


class TestLoadSplit:
    """Test end-to-end data loading via load_split."""

    def test_load_with_expansion(self, temp_json_files):
        print_separator("TEST: Load Dataset With Expansion")

        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
        )

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3  # 1 original + type1 + type2
        assert "conversations" in dataset.features
        assert "claim_id" in dataset.features
        assert "perturbation_type" in dataset.features
        print("✓ Dataset loaded and expanded successfully")

    def test_load_without_expansion(self, temp_json_files):
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=False,
        )

        assert len(dataset) == 1
        assert all(
            p == "original_normalized" for p in dataset["perturbation_type"]
        )

    def test_load_multiple_files(self, temp_json_files):
        dataset = load_split(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations=True,
        )

        claim_ids = set(dataset["claim_id"])
        assert len(claim_ids) == 2
        # Each file has 1 example → 3 expanded each = 6 total
        assert len(dataset) == 6

    def test_load_with_file_configs(self, temp_json_files):
        """Load with per-file type control via FileLoadConfig."""
        print_separator("TEST: Load With FileLoadConfig")

        configs = [
            FileLoadConfig(
                path=temp_json_files["train_1"],
                perturb_types=[
                    PerturbationType.PERTURB_TYPE1,
                    PerturbationType.PERTURB_TYPE2,
                ],
            ),
            FileLoadConfig(
                path=temp_json_files["train_2"],
                perturb_types=[PerturbationType.PERTURB_TYPE1],  # skip type2
            ),
        ]

        dataset = load_split(
            file_configs=configs,
            split="train",
            expand_perturbations=True,
        )

        print(f"Dataset size: {len(dataset)}")
        perturb_dist = Counter(dataset["perturbation_type"])
        for pt, count in perturb_dist.most_common():
            print(f"  {pt}: {count}")

        # File 1: original + type1 + type2 = 3
        # File 2: original + type1 (NOT type2) = 2
        assert len(dataset) == 5
        assert perturb_dist["original_normalized"] == 2
        assert perturb_dist["perturb_type1"] == 2
        assert perturb_dist["perturb_type2"] == 1  # only from file 1

        print("✓ Per-file type control works")

    def test_file_configs_type1_only_from_false_file(self, temp_json_files):
        """Load only type1 from the False file; all results must be False."""
        print_separator("TEST: FileLoadConfig type1-only from False file")

        configs = [
            FileLoadConfig(
                path=temp_json_files["train_2"],  # approximation_False.json
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
        ]

        dataset = load_split(
            file_configs=configs,
            split="train",
            expand_perturbations=True,
        )

        print(f"Dataset size: {len(dataset)}")
        for i, ex in enumerate(dataset):
            print(
                f"  [{i}] {ex['perturbation_type']:25s} "
                f"label={ex['label']}  claim_id={ex['claim_id']}"
            )

        # original + type1
        assert len(dataset) == 2

        perturb_dist = Counter(dataset["perturbation_type"])
        assert perturb_dist["original_normalized"] == 1
        assert perturb_dist["perturb_type1"] == 1
        assert "perturb_type2" not in perturb_dist

        # All labels must be False (type2 is always False regardless)
        assert all(
            lbl == '{"label": false}' for lbl in dataset["label"]
        ), "type1-only from False file must produce only False labels"

        print("✓ type1-only from False file yields only False labels")

    def test_file_configs_mixed_selective_loading(self, temp_json_files):
        """Load all types from True file but only type1 from False file."""
        print_separator("TEST: Mixed selective loading across files")

        configs = [
            FileLoadConfig(
                path=temp_json_files["train_1"],  # numeration_True.json
                perturb_types=None,  # all types
            ),
            FileLoadConfig(
                path=temp_json_files["train_2"],  # approximation_False.json
                perturb_types=[PerturbationType.PERTURB_TYPE1],  # skip type2
            ),
        ]

        dataset = load_split(
            file_configs=configs,
            split="train",
            expand_perturbations=True,
        )

        print(f"Dataset size: {len(dataset)}")
        perturb_dist = Counter(dataset["perturbation_type"])
        label_dist = Counter(dataset["label"])
        for pt, count in perturb_dist.most_common():
            print(f"  {pt}: {count}")
        print(f"  Labels: {dict(label_dist)}")

        # True file: original + type1 + type2 = 3
        # False file: original + type1 = 2
        assert len(dataset) == 5
        assert perturb_dist["original_normalized"] == 2
        assert perturb_dist["perturb_type1"] == 2
        assert perturb_dist["perturb_type2"] == 1  # only from True file

        # Label distribution:
        # True file: original(True) + type1(True) + type2(False) = 2T + 1F
        # False file: original(False) + type1(False) = 2F
        assert label_dist['{"label": true}'] == 2
        assert label_dist['{"label": false}'] == 3

        # Verify type2 only came from the True file (claim_id = train_000001)
        type2_entries = [
            ex for ex in dataset if ex["perturbation_type"] == "perturb_type2"
        ]
        assert len(type2_entries) == 1
        assert type2_entries[0]["claim_id"] == "train_000001"
        # type2 is always False
        assert type2_entries[0]["label"] == '{"label": false}'

        print("✓ Mixed selective loading correct")

    def test_file_configs_load_all_when_none(self, temp_json_files):
        """FileLoadConfig with perturb_types=None loads everything."""
        configs = [FileLoadConfig(path=temp_json_files["train_1"])]
        dataset = load_split(
            file_configs=configs,
            split="train",
            expand_perturbations=True,
        )
        assert len(dataset) == 3  # original + type1 + type2

    def test_no_paths_or_configs_raises(self):
        with pytest.raises(
            ValueError, match="at least one file path must be provided"
        ):
            load_split()

    def test_load_train_and_val(self, temp_json_files):
        """Load train and validation splits separately."""
        print_separator("TEST: Load Train and Validation")

        train_ds = load_split(
            json_paths=[
                temp_json_files["train_1"],
                temp_json_files["train_2"],
            ],
            split="train",
            expand_perturbations=True,
        )
        val_ds = load_split(
            json_paths=temp_json_files["val"],
            split="validation",
            expand_perturbations=True,
        )

        assert len(train_ds) == 6  # 2 files × 3 expanded each
        assert len(val_ds) == 3

        print("✓ Train and validation loaded correctly")

    def test_load_train_with_configs_val_with_path(self, temp_json_files):
        """Use file_configs for train, simple path for val."""
        print_separator("TEST: Load Train/Val with FileLoadConfig")

        train_configs = [
            FileLoadConfig(
                path=temp_json_files["train_1"],
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
        ]

        train_ds = load_split(
            file_configs=train_configs,
            split="train",
            expand_perturbations=True,
        )
        val_ds = load_split(
            json_paths=temp_json_files["val"],
            split="validation",
            expand_perturbations=True,
        )

        assert len(train_ds) == 2  # original + type1 only
        assert len(val_ds) == 3  # all types

        print("✓ Mixed config/path loading works")

    def test_empty_perturbation_value_skipped(self, tmp_path):
        """Perturbation keys with empty/null values should be skipped."""
        example = {
            "claim_id": "test_001",
            "label": True,
            "normalized_claim": "Test claim.",
            "claim": "Test claim.",
            "doc": "Test doc.",
            "perturb_type1_True": "",
            "perturb_type2_True": None,
        }
        f = tmp_path / "empty_perturb.json"
        f.write_text(json.dumps([example]))

        dataset = load_split(json_paths=str(f), split="train")
        assert len(dataset) == 1  # only original


# ---------------------------------------------------------------------------
# Tests — conversation format
# ---------------------------------------------------------------------------


class TestConversationFormat:
    """Test conversation format produced by load_split."""

    def test_train_conversation_format(self, temp_json_files):
        print_separator("TEST: Train Conversation Format")

        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
        )

        first = dataset[0]
        assert "conversations" in first
        assert "claim_id" in first
        assert "perturbation_type" in first
        assert "label" in first

        conv = first["conversations"]
        assert len(conv) == 3  # system, user, assistant
        assert conv[0]["role"] == "system"
        assert conv[1]["role"] == "user"
        assert conv[2]["role"] == "assistant"
        assert conv[2]["content"] in ['{"label": true}', '{"label": false}']

        print("✓ Train conversation format correct")

    def test_conversation_includes_evidence(self, temp_json_files):
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
            include_doc=True,
        )
        user_msg = dataset[0]["conversations"][1]["content"]
        assert "35,000 LED bulbs" in user_msg or "LED bulbs" in user_msg

    def test_conversation_excludes_evidence(self, temp_json_files):
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
            include_doc=False,
        )
        user_msg = dataset[0]["conversations"][1]["content"]
        # Evidence text should not appear when include_doc=False
        assert "Official report shows" not in user_msg

    def test_label_json_format(self, temp_json_files):
        """Label should be JSON-formatted."""
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
        )

        orig = [
            ex
            for ex in dataset
            if ex["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == '{"label": true}'

        t2 = [
            ex for ex in dataset if ex["perturbation_type"] == "perturb_type2"
        ][0]
        assert t2["label"] == '{"label": false}'


# ---------------------------------------------------------------------------
# Tests — perturbation type distribution
# ---------------------------------------------------------------------------


class TestPerturbationTypeDistribution:
    """Test perturbation type distribution analysis."""

    def test_perturbation_distribution(self, temp_json_files):
        print_separator("TEST: Perturbation Type Distribution")

        dataset = load_split(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations=True,
        )

        perturb_counts = Counter(dataset["perturbation_type"])

        print(f"Total examples: {len(dataset)}")
        for pt, count in sorted(perturb_counts.items()):
            pct = 100 * count / len(dataset)
            print(f"  {pt:30s}: {count:3d} ({pct:5.1f}%)")

        assert "original_normalized" in perturb_counts
        assert perturb_counts["original_normalized"] == 2

        print("✓ Distribution analysis complete")


# ---------------------------------------------------------------------------
# Tests — claim_id tracking
# ---------------------------------------------------------------------------


class TestClaimIdTracking:
    """Test claim_id tracking across perturbations."""

    def test_claim_id_consistency(self, temp_json_files):
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
        )

        by_claim_id = defaultdict(list)
        for example in dataset:
            by_claim_id[example["claim_id"]].append(
                example["perturbation_type"]
            )

        # Single file = single claim_id
        assert len(by_claim_id) == 1
        claim_id = list(by_claim_id.keys())[0]
        assert claim_id == "train_000001"
        assert "original_normalized" in by_claim_id[claim_id]
        assert "perturb_type1" in by_claim_id[claim_id]
        assert "perturb_type2" in by_claim_id[claim_id]


# ---------------------------------------------------------------------------
# Tests — boolean label handling
# ---------------------------------------------------------------------------


class TestBooleanLabelHandling:
    """Verify that bool labels from JSON (True/False) are handled correctly."""

    def test_bool_true_label_becomes_json(self, temp_json_files):
        """Bool True label in JSON should become '{"label": true}' string."""
        dataset = load_split(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations=True,
        )
        orig = [
            ex
            for ex in dataset
            if ex["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == '{"label": true}'
        assert isinstance(orig["label"], str)

    def test_bool_false_label_becomes_json(self, temp_json_files):
        """Bool False label in JSON should become '{"label": false}' string."""
        dataset = load_split(
            json_paths=temp_json_files["train_2"],
            split="train",
            expand_perturbations=True,
        )
        orig = [
            ex
            for ex in dataset
            if ex["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == '{"label": false}'
        assert isinstance(orig["label"], str)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING DATA LOADER TESTS")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])
