import pytest
import json
from typing import Dict
from collections import Counter, defaultdict
from datasets import Dataset

from src.llm_finetune.data_loading_utils.hf_dataset_instructions import (
    load_json_files,
    expand_perturbations,
    convert_to_conversations,
    load_fact_checking_data,
    load_train_and_validation,
    PerturbationType,
    FileLoadConfig,
    parse_file_config,
    _detect_perturb_key,
    _normalize_label,
    _determine_perturbed_label,
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
# Tests — internal helpers
# ---------------------------------------------------------------------------


class TestDetectPerturbKey:
    """Test the regex-based perturbation key detection."""

    def test_type1_keys(self):
        assert _detect_perturb_key("perturb_type1_True") == "perturb_type1"
        assert _detect_perturb_key("perturb_type1_False") == "perturb_type1"

    def test_type2_keys(self):
        assert _detect_perturb_key("perturb_type2_True") == "perturb_type2"
        assert _detect_perturb_key("perturb_type2_False") == "perturb_type2"

    def test_type3_keys_with_label_suffix(self):
        assert (
            _detect_perturb_key("perturb_type3_random_False")
            == "perturb_type3_random"
        )
        assert (
            _detect_perturb_key("perturb_type3_masked_True")
            == "perturb_type3_masked"
        )
        assert (
            _detect_perturb_key("perturb_type3_removed_False")
            == "perturb_type3_removed"
        )

    def test_type3_keys_without_label_suffix(self):
        assert (
            _detect_perturb_key("perturb_type3_random")
            == "perturb_type3_random"
        )
        assert (
            _detect_perturb_key("perturb_type3_masked")
            == "perturb_type3_masked"
        )
        assert (
            _detect_perturb_key("perturb_type3_removed")
            == "perturb_type3_removed"
        )

    def test_non_perturb_keys_return_none(self):
        assert _detect_perturb_key("claim_id") is None
        assert _detect_perturb_key("label") is None
        assert _detect_perturb_key("normalized_claim") is None
        assert _detect_perturb_key("doc") is None


class TestNormalizeLabel:
    """Test label normalization (bool/string → 'True'/'False')."""

    def test_bool_labels(self):
        assert _normalize_label(True) == "True"
        assert _normalize_label(False) == "False"

    def test_string_labels(self):
        assert _normalize_label("True") == "True"
        assert _normalize_label("False") == "False"
        assert _normalize_label("true") == "True"
        assert _normalize_label("false") == "False"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            _normalize_label("maybe")
        with pytest.raises(ValueError):
            _normalize_label(42)


class TestDetermineLabel:
    """Test perturbation label determination logic."""

    def test_type1_preserves_label(self):
        assert _determine_perturbed_label("perturb_type1", True) == "True"
        assert _determine_perturbed_label("perturb_type1", False) == "False"

    def test_type2_flips_label(self):
        assert _determine_perturbed_label("perturb_type2", True) == "False"
        assert _determine_perturbed_label("perturb_type2", False) == "True"

    def test_type3_always_false(self):
        for t3 in [
            "perturb_type3_random",
            "perturb_type3_masked",
            "perturb_type3_removed",
        ]:
            assert _determine_perturbed_label(t3, True) == "False"
            assert _determine_perturbed_label(t3, False) == "False"


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
# Tests — load_json_files
# ---------------------------------------------------------------------------


class TestLoadJsonFiles:
    """Test JSON file loading functionality."""

    def test_load_single_json_file(self, temp_json_files):
        print_separator("TEST: Load Single JSON File")
        examples = load_json_files(temp_json_files["train_1"])

        assert len(examples) == 1
        assert examples[0]["claim_id"] == "train_000001"
        print("✓ Single file loading successful")

    def test_load_multiple_json_files(self, temp_json_files):
        print_separator("TEST: Load Multiple JSON Files")
        examples = load_json_files(
            [temp_json_files["train_1"], temp_json_files["train_2"]]
        )

        assert len(examples) == 2
        assert examples[0]["claim_id"] == "train_000001"
        assert examples[1]["claim_id"] == "train_000002"
        print("✓ Multiple file loading successful")


# ---------------------------------------------------------------------------
# Tests — expand_perturbations
# ---------------------------------------------------------------------------


class TestExpandPerturbations:
    """Test perturbation expansion with the new key format."""

    def test_expand_numeration_true(self, numeration_true_example):
        """numeration_True.json: original True + type1 (True) + type2
        (False)."""
        print_separator("TEST: Expand numeration True example")

        expanded = expand_perturbations([numeration_true_example])

        print(f"Expanded {len(expanded)} examples:")
        for ex in expanded:
            print(f"  {ex['perturbation_type']:25s} label={ex['label']}")

        assert len(expanded) == 3  # original + type1 + type2

        orig = [
            e
            for e in expanded
            if e["perturbation_type"] == "original_normalized"
        ]
        assert len(orig) == 1
        assert orig[0]["label"] == "True"

        t1 = [e for e in expanded if e["perturbation_type"] == "perturb_type1"]
        assert len(t1) == 1
        assert t1[0]["label"] == "True"  # structure-preserving

        t2 = [e for e in expanded if e["perturbation_type"] == "perturb_type2"]
        assert len(t2) == 1
        assert t2[0]["label"] == "False"  # value-modifying → flipped

    def test_expand_approximation_false(self, approximation_false_example):
        """approximation_False.json: original False + type1 (False) + type2
        (True)."""
        print_separator("TEST: Expand approximation False example")

        expanded = expand_perturbations([approximation_false_example])

        print(f"Expanded {len(expanded)} examples:")
        for ex in expanded:
            print(f"  {ex['perturbation_type']:25s} label={ex['label']}")

        assert len(expanded) == 3

        orig = [
            e
            for e in expanded
            if e["perturbation_type"] == "original_normalized"
        ]
        assert orig[0]["label"] == "False"

        t1 = [e for e in expanded if e["perturbation_type"] == "perturb_type1"]
        assert t1[0]["label"] == "False"  # structure-preserving

        t2 = [e for e in expanded if e["perturbation_type"] == "perturb_type2"]
        assert t2[0]["label"] == "True"  # value-modifying → flipped

    def test_expand_obfuscation_true(self, obfuscation_true_example):
        """obfuscation_True.json: original True + 3 type3 variants (all
        False)."""
        print_separator("TEST: Expand obfuscation True example")

        expanded = expand_perturbations([obfuscation_true_example])

        print(f"Expanded {len(expanded)} examples:")
        for ex in expanded:
            print(f"  {ex['perturbation_type']:25s} label={ex['label']}")

        assert len(expanded) == 4  # original + 3 type3 variants

        # All type3 perturbations should be False
        type3 = [
            e
            for e in expanded
            if e["perturbation_type"].startswith("perturb_type3")
        ]
        assert len(type3) == 3
        assert all(e["label"] == "False" for e in type3)

    def test_expand_negative_only_type2(self, negative_true_example):
        """negative_True.json: only has type2 (no type1)."""
        expanded = expand_perturbations([negative_true_example])

        assert len(expanded) == 2  # original + type2
        t2 = [e for e in expanded if e["perturbation_type"] == "perturb_type2"]
        assert len(t2) == 1
        assert t2[0]["label"] == "False"  # flipped from True

    def test_allowed_types_filter(self, numeration_true_example):
        """Only load type1, skip type2."""
        print_separator("TEST: allowed_types filter")

        expanded = expand_perturbations(
            [numeration_true_example],
            allowed_types=[PerturbationType.PERTURB_TYPE1],
        )

        print(f"Expanded {len(expanded)} examples (type1 only):")
        for ex in expanded:
            print(f"  {ex['perturbation_type']:25s} label={ex['label']}")

        assert len(expanded) == 2  # original + type1 only
        types = {e["perturbation_type"] for e in expanded}
        assert "perturb_type2" not in types
        assert "original_normalized" in types
        assert "perturb_type1" in types

    def test_allowed_types_filter_type2_only(self, approximation_false_example):
        """Only load type2 from a False example."""
        expanded = expand_perturbations(
            [approximation_false_example],
            allowed_types=[PerturbationType.PERTURB_TYPE2],
        )

        assert len(expanded) == 2  # original + type2
        types = {e["perturbation_type"] for e in expanded}
        assert "perturb_type1" not in types
        assert "perturb_type2" in types

    def test_selective_type1_only_from_false_excludes_type2(
        self, approximation_false_example
    ):
        """Load only type1 from a False-label example; type2 must be excluded.

        For a False dataset:
        - type1 (structure-preserving) keeps the False label
        - type2 (value-modifying) would flip to True
        Selecting only type1 should therefore produce *only* False labels.
        """
        print_separator("TEST: Selective type1 from False dataset (no type2)")

        expanded = expand_perturbations(
            [approximation_false_example],
            allowed_types=[PerturbationType.PERTURB_TYPE1],
        )

        print(f"Expanded {len(expanded)} examples (type1 only from False):")
        for ex in expanded:
            print(f"  {ex['perturbation_type']:25s} label={ex['label']}")

        # Should contain original_normalized + type1, NOT type2
        assert len(expanded) == 2
        types = {e["perturbation_type"] for e in expanded}
        assert types == {"original_normalized", "perturb_type1"}
        assert "perturb_type2" not in types

        # All labels must be False (type2 would have flipped to True)
        assert all(
            e["label"] == "False" for e in expanded
        ), "Excluding type2 from a False file should yield only False labels"

        # Verify claim content is from the False example
        for ex in expanded:
            assert ex["claim_id"] == "train_000002"

        print("✓ type1-only from False dataset produces only False labels")

    def test_selective_type1_from_false_vs_all_types(
        self, approximation_false_example
    ):
        """Compare type1-only vs all-types from a False-label example.

        Loading all types introduces a True-labeled entry (type2); filtering to
        type1-only must eliminate it.
        """
        all_expanded = expand_perturbations([approximation_false_example])
        type1_only = expand_perturbations(
            [approximation_false_example],
            allowed_types=[PerturbationType.PERTURB_TYPE1],
        )

        # All types: original(False) + type1(False) + type2(True) = 3
        assert len(all_expanded) == 3
        all_labels = {e["label"] for e in all_expanded}
        assert all_labels == {"True", "False"}  # both labels present

        # Type1 only: original(False) + type1(False) = 2
        assert len(type1_only) == 2
        type1_labels = {e["label"] for e in type1_only}
        assert type1_labels == {"False"}  # only False labels

    def test_allowed_types_none_loads_all(self, numeration_true_example):
        """allowed_types=None should load all available perturbations."""
        expanded = expand_perturbations(
            [numeration_true_example], allowed_types=None
        )
        assert len(expanded) == 3  # original + type1 + type2

    def test_empty_perturbation_value_skipped(self):
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
        expanded = expand_perturbations([example])
        assert len(expanded) == 1  # only original


# ---------------------------------------------------------------------------
# Tests — convert_to_conversations
# ---------------------------------------------------------------------------


class TestConvertToConversations:
    """Test conversation format conversion."""

    def test_train_conversation_format(self, numeration_true_example):
        print_separator("TEST: Train Conversation Format")

        expanded = expand_perturbations([numeration_true_example])
        conversations = convert_to_conversations(expanded, split="train")

        assert len(conversations) == len(expanded)

        first = conversations[0]
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

    def test_conversation_includes_evidence(self, numeration_true_example):
        expanded = expand_perturbations([numeration_true_example])
        conversations = convert_to_conversations(
            expanded, split="train", include_doc=True
        )
        user_msg = conversations[0]["conversations"][1]["content"]
        assert numeration_true_example["doc"] in user_msg

    def test_conversation_excludes_evidence(self, numeration_true_example):
        expanded = expand_perturbations([numeration_true_example])
        conversations = convert_to_conversations(
            expanded, split="train", include_doc=False
        )
        user_msg = conversations[0]["conversations"][1]["content"]
        assert numeration_true_example["doc"] not in user_msg

    def test_label_json_format(self, numeration_true_example):
        """Label should be JSON-formatted."""
        expanded = expand_perturbations([numeration_true_example])
        conversations = convert_to_conversations(expanded, split="train")

        orig = [
            c
            for c in conversations
            if c["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == '{"label": true}'

        t2 = [
            c
            for c in conversations
            if c["perturbation_type"] == "perturb_type2"
        ][0]
        assert t2["label"] == '{"label": false}'


# ---------------------------------------------------------------------------
# Tests — load_fact_checking_data (end-to-end)
# ---------------------------------------------------------------------------


class TestLoadFactCheckingData:
    """Test end-to-end data loading."""

    def test_load_with_expansion(self, temp_json_files):
        print_separator("TEST: Load Dataset With Expansion")

        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=True,
        )

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3  # 1 original + type1 + type2
        assert "conversations" in dataset.features
        assert "claim_id" in dataset.features
        assert "perturbation_type" in dataset.features
        print("✓ Dataset loaded and expanded successfully")

    def test_load_without_expansion(self, temp_json_files):
        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=False,
        )

        assert len(dataset) == 1
        assert all(
            p == "original_normalized" for p in dataset["perturbation_type"]
        )

    def test_load_multiple_files(self, temp_json_files):
        dataset = load_fact_checking_data(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations_flag=True,
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

        dataset = load_fact_checking_data(
            file_configs=configs,
            split="train",
            expand_perturbations_flag=True,
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

        dataset = load_fact_checking_data(
            file_configs=configs,
            split="train",
            expand_perturbations_flag=True,
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

        # All labels must be False (type2 would have introduced True)
        assert all(
            lbl == '{"label": false}' for lbl in dataset["label"]
        ), "type1-only from False file must produce only False labels"

        print("✓ type1-only from False file yields only False labels")

    def test_file_configs_mixed_selective_loading(self, temp_json_files):
        """Load all types from True file but only type1 from False file.

        Verifies per-file control: True file contributes type1+type2,
        False file contributes only type1 (not the label-flipping type2).
        """
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

        dataset = load_fact_checking_data(
            file_configs=configs,
            split="train",
            expand_perturbations_flag=True,
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
        # type2 from True file flips label → False
        assert type2_entries[0]["label"] == '{"label": false}'

        print("✓ Mixed selective loading correct")

    def test_file_configs_load_all_when_none(self, temp_json_files):
        """FileLoadConfig with perturb_types=None loads everything."""
        configs = [FileLoadConfig(path=temp_json_files["train_1"])]
        dataset = load_fact_checking_data(
            file_configs=configs,
            split="train",
            expand_perturbations_flag=True,
        )
        assert len(dataset) == 3  # original + type1 + type2

    def test_no_paths_or_configs_raises(self):
        with pytest.raises(
            ValueError, match="Either json_paths or file_configs"
        ):
            load_fact_checking_data()


# ---------------------------------------------------------------------------
# Tests — load_train_and_validation
# ---------------------------------------------------------------------------


class TestLoadTrainAndValidation:
    """Test train/validation split loading."""

    def test_load_train_and_val_simple(self, temp_json_files):
        print_separator("TEST: Load Train and Validation (simple paths)")

        datasets = load_train_and_validation(
            train_json_paths=[
                temp_json_files["train_1"],
                temp_json_files["train_2"],
            ],
            val_json_paths=temp_json_files["val"],
            expand_perturbations_flag=True,
        )

        assert "train" in datasets
        assert "validation" in datasets
        assert len(datasets["train"]) == 6  # 2 files × 3 expanded each
        assert len(datasets["validation"]) == 3

        print("✓ Train and validation loaded correctly")

    def test_load_train_and_val_with_configs(self, temp_json_files):
        """Use file_configs for train, simple path for val."""
        print_separator("TEST: Load Train/Val with FileLoadConfig")

        train_configs = [
            FileLoadConfig(
                path=temp_json_files["train_1"],
                perturb_types=[PerturbationType.PERTURB_TYPE1],
            ),
        ]

        datasets = load_train_and_validation(
            train_file_configs=train_configs,
            val_json_paths=temp_json_files["val"],
            expand_perturbations_flag=True,
        )

        assert len(datasets["train"]) == 2  # original + type1 only
        assert len(datasets["validation"]) == 3  # all types

        print("✓ Mixed config/path loading works")


# ---------------------------------------------------------------------------
# Tests — perturbation type distribution
# ---------------------------------------------------------------------------


class TestPerturbationTypeDistribution:
    """Test perturbation type distribution analysis."""

    def test_perturbation_distribution(self, temp_json_files):
        print_separator("TEST: Perturbation Type Distribution")

        dataset = load_fact_checking_data(
            json_paths=[temp_json_files["train_1"], temp_json_files["train_2"]],
            split="train",
            expand_perturbations_flag=True,
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
        dataset = load_fact_checking_data(
            json_paths=temp_json_files["train_1"],
            split="train",
            expand_perturbations_flag=True,
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

    def test_bool_true_label_becomes_string(self, numeration_true_example):
        expanded = expand_perturbations([numeration_true_example])
        orig = [
            e
            for e in expanded
            if e["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == "True"
        assert isinstance(orig["label"], str)

    def test_bool_false_label_becomes_string(self, approximation_false_example):
        expanded = expand_perturbations([approximation_false_example])
        orig = [
            e
            for e in expanded
            if e["perturbation_type"] == "original_normalized"
        ][0]
        assert orig["label"] == "False"
        assert isinstance(orig["label"], str)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RUNNING DATA LOADER TESTS")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])
