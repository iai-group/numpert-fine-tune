"""Tests for the training_config_export utility."""

import json
import pytest
from datetime import datetime

from datasets import Dataset

from src.llm_finetune.training_config_export import (
    CONFIG_FILENAME,
    _dataset_stats,
    _file_config_to_dict,
    build_training_config,
    save_training_config,
)
from src.llm_finetune.data_loading_utils.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

FIXED_TIMESTAMP = datetime(2025, 6, 15, 12, 0, 0)


@pytest.fixture
def sample_train_configs():
    """Return a list of FileLoadConfig for training files."""
    return [
        FileLoadConfig(
            path="data/perturbed_data/train_split/approximation_True.json",
            perturb_types=[
                PerturbationType.PERTURB_TYPE1,
                PerturbationType.PERTURB_TYPE2,
            ],
        ),
        FileLoadConfig(
            path="data/perturbed_data/train_split/numeration_False.json",
            perturb_types=None,  # all types
        ),
    ]


@pytest.fixture
def sample_val_configs():
    """Return a list of FileLoadConfig for validation files."""
    return [
        FileLoadConfig(
            path="data/perturbed_data/val_split/approximation_True.json",
            perturb_types=[PerturbationType.PERTURB_TYPE1],
        ),
    ]


def _make_dataset(
    n: int = 10,
    perturb_types=None,
    labels=None,
) -> Dataset:
    """Create a minimal HuggingFace dataset for testing.

    Args:
        n: Number of examples.
        perturb_types: Perturbation type values to cycle through.
        labels: Label values to cycle through.

    Returns:
        A HuggingFace Dataset with required columns.
    """
    if perturb_types is None:
        perturb_types = [
            "original_normalized",
            "perturb_type1",
            "perturb_type2",
        ]
    if labels is None:
        labels = ["True", "False"]

    return Dataset.from_dict(
        {
            "claim_id": [f"claim_{i}" for i in range(n)],
            "perturbation_type": [
                perturb_types[i % len(perturb_types)] for i in range(n)
            ],
            "label": [labels[i % len(labels)] for i in range(n)],
            "claim": [f"Claim text {i}" for i in range(n)],
            "doc": [f"Evidence text {i}" for i in range(n)],
        }
    )


# ------------------------------------------------------------------
# _file_config_to_dict
# ------------------------------------------------------------------


class TestFileConfigToDict:
    """Tests for _file_config_to_dict."""

    def test_with_specific_types(self):
        """Specific perturb types are serialized as value strings."""
        config = FileLoadConfig(
            path="train/approx_True.json",
            perturb_types=[
                PerturbationType.PERTURB_TYPE1,
                PerturbationType.PERTURB_TYPE3_MASKED,
            ],
        )
        result = _file_config_to_dict(config)

        assert result["path"] == "train/approx_True.json"
        assert result["perturb_types"] == [
            "perturb_type1",
            "perturb_type3_masked",
        ]

    def test_with_all_types(self):
        """perturb_types=None is preserved as None (meaning all)."""
        config = FileLoadConfig(
            path="train/numeration_False.json",
            perturb_types=None,
        )
        result = _file_config_to_dict(config)

        assert result["path"] == "train/numeration_False.json"
        assert result["perturb_types"] is None

    def test_empty_types_list(self):
        """An empty list serializes as an empty list."""
        config = FileLoadConfig(path="empty.json", perturb_types=[])
        result = _file_config_to_dict(config)

        assert result["perturb_types"] == []


# ------------------------------------------------------------------
# _dataset_stats
# ------------------------------------------------------------------


class TestDatasetStats:
    """Tests for _dataset_stats."""

    def test_basic_distribution(self):
        """Perturbation and label distributions are computed correctly."""
        ds = _make_dataset(
            n=6,
            perturb_types=["original_normalized", "perturb_type1"],
            labels=["True", "False"],
        )
        stats = _dataset_stats(ds)

        assert stats["size"] == 6
        assert stats["perturbation_distribution"]["original_normalized"] == 3
        assert stats["perturbation_distribution"]["perturb_type1"] == 3
        assert stats["label_distribution"]["True"] == 3
        assert stats["label_distribution"]["False"] == 3

    def test_empty_dataset(self):
        """An empty dataset returns zeroed stats."""
        ds = _make_dataset(n=0, perturb_types=["x"], labels=["True"])
        stats = _dataset_stats(ds)

        assert stats["size"] == 0
        assert stats["perturbation_distribution"] == {}
        assert stats["label_distribution"] == {}

    def test_single_example(self):
        """A single-example dataset produces correct counts."""
        ds = _make_dataset(
            n=1,
            perturb_types=["perturb_type2"],
            labels=["False"],
        )
        stats = _dataset_stats(ds)

        assert stats["size"] == 1
        assert stats["perturbation_distribution"] == {"perturb_type2": 1}
        assert stats["label_distribution"] == {"False": 1}


# ------------------------------------------------------------------
# build_training_config
# ------------------------------------------------------------------


class TestBuildTrainingConfig:
    """Tests for build_training_config."""

    def test_minimal_config(self):
        """Builds a valid config with only the required argument."""
        config = build_training_config(
            model_name="unsloth/Qwen3-8B",
            timestamp=FIXED_TIMESTAMP,
        )

        assert config["model"]["name"] == "unsloth/Qwen3-8B"
        assert config["timestamp"] == FIXED_TIMESTAMP.isoformat()
        assert config["data"]["train_files"] == []
        assert config["data"]["val_files"] == []
        # No dataset stats when datasets are not provided
        assert "train_stats" not in config["data"]
        assert "val_stats" not in config["data"]

    def test_full_config(self, sample_train_configs, sample_val_configs):
        """All parameters are recorded in the output dictionary."""
        train_ds = _make_dataset(n=20)
        val_ds = _make_dataset(n=5)

        config = build_training_config(
            model_name="unsloth/Qwen3-32B",
            train_file_configs=sample_train_configs,
            val_file_configs=sample_val_configs,
            train_dataset=train_ds,
            val_dataset=val_ds,
            rank=64,
            num_train_epochs=3,
            start_epoch=2,
            batch_size=4,
            gradient_accumulation_steps=8,
            max_seq_length=8192,
            learning_rate=2e-5,
            expand_perturbations=True,
            balance_labels=True,
            perturb_ratio=0.5,
            false_perturb_t_to_f_ratio=0.33,
            batch_match_claim_id=True,
            split_large_claims=False,
            ensure_original_in_batch=False,
            ensure_perturbed_in_batch=True,
            save_gguf=True,
            gguf_quantization_method="q4_k_m",
            timestamp=FIXED_TIMESTAMP,
        )

        # Model section
        assert config["model"]["name"] == "unsloth/Qwen3-32B"
        assert config["model"]["lora_rank"] == 64

        # Training section
        t = config["training"]
        assert t["start_epoch"] == 2
        assert t["num_train_epochs"] == 3
        assert t["batch_size"] == 4
        assert t["gradient_accumulation_steps"] == 8
        assert t["effective_batch_size"] == 32
        assert t["max_seq_length"] == 8192
        assert t["learning_rate"] == 2e-5

        # Data section - file configs
        d = config["data"]
        assert len(d["train_files"]) == 2
        assert d["train_files"][0]["perturb_types"] == [
            "perturb_type1",
            "perturb_type2",
        ]
        assert d["train_files"][1]["perturb_types"] is None
        assert len(d["val_files"]) == 1

        # Data section - flags
        assert d["expand_perturbations"] is True
        assert d["balance_labels"] is True
        assert d["perturb_ratio"] == 0.5
        assert d["false_perturb_t_to_f_ratio"] == 0.33

        # Dataset stats
        assert d["train_stats"]["size"] == 20
        assert d["val_stats"]["size"] == 5

        # Batch sampler section
        bs = config["batch_sampler"]
        assert bs["batch_match_claim_id"] is True
        assert bs["split_large_claims"] is False
        assert bs["ensure_original_in_batch"] is False
        assert bs["ensure_perturbed_in_batch"] is True

        # GGUF section
        assert config["gguf"]["save_gguf"] is True
        assert config["gguf"]["quantization_method"] == "q4_k_m"

    def test_effective_batch_size_calculation(self):
        """Effective batch size = batch_size * gradient_accumulation."""
        config = build_training_config(
            model_name="test",
            batch_size=8,
            gradient_accumulation_steps=4,
            timestamp=FIXED_TIMESTAMP,
        )
        assert config["training"]["effective_batch_size"] == 32

    def test_default_timestamp_is_generated(self):
        """When no timestamp is given, a current one is generated."""
        config = build_training_config(model_name="test")
        # Should be a valid ISO-format string
        parsed = datetime.fromisoformat(config["timestamp"])
        assert isinstance(parsed, datetime)

    def test_datasets_omitted_when_none(self):
        """Dataset stats are excluded when datasets are not provided."""
        config = build_training_config(
            model_name="test",
            train_dataset=None,
            val_dataset=None,
            timestamp=FIXED_TIMESTAMP,
        )
        assert "train_stats" not in config["data"]
        assert "val_stats" not in config["data"]

    def test_config_is_json_serializable(self, sample_train_configs):
        """The returned dict can be serialized to JSON without errors."""
        config = build_training_config(
            model_name="test",
            train_file_configs=sample_train_configs,
            train_dataset=_make_dataset(n=4),
            timestamp=FIXED_TIMESTAMP,
        )
        serialized = json.dumps(config)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped == config


# ------------------------------------------------------------------
# save_training_config
# ------------------------------------------------------------------


class TestSaveTrainingConfig:
    """Tests for save_training_config."""

    def test_creates_file(self, tmp_path):
        """JSON file is created at the expected location."""
        config = build_training_config(
            model_name="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_training_config(config, tmp_path)

        assert result_path.exists()
        assert result_path.name == CONFIG_FILENAME
        assert result_path.parent == tmp_path

    def test_file_content_matches(self, tmp_path):
        """Written JSON matches the original dictionary."""
        config = build_training_config(
            model_name="unsloth/Qwen3-8B",
            rank=16,
            batch_size=8,
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_training_config(config, tmp_path)

        with open(result_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == config

    def test_creates_nested_directories(self, tmp_path):
        """Missing parent directories are created automatically."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        config = build_training_config(
            model_name="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_training_config(config, nested_dir)

        assert result_path.exists()
        assert result_path.parent == nested_dir

    def test_custom_filename(self, tmp_path):
        """A custom filename can be specified."""
        config = build_training_config(
            model_name="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_training_config(
            config, tmp_path, filename="my_config.json"
        )

        assert result_path.name == "my_config.json"
        assert result_path.exists()

    def test_full_roundtrip(self, tmp_path, sample_train_configs):
        """Full roundtrip: build -> save -> load produces identical data."""
        train_ds = _make_dataset(n=12)

        config = build_training_config(
            model_name="unsloth/Qwen3-32B",
            train_file_configs=sample_train_configs,
            train_dataset=train_ds,
            rank=64,
            num_train_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_seq_length=4096,
            batch_match_claim_id=True,
            timestamp=FIXED_TIMESTAMP,
        )

        result_path = save_training_config(config, tmp_path)

        with open(result_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Verify key sections survived the roundtrip
        assert loaded["model"]["name"] == "unsloth/Qwen3-32B"
        assert loaded["model"]["lora_rank"] == 64
        assert loaded["training"]["num_train_epochs"] == 5
        assert loaded["training"]["effective_batch_size"] == 8
        assert len(loaded["data"]["train_files"]) == 2
        assert loaded["data"]["train_stats"]["size"] == 12
        assert loaded["batch_sampler"]["batch_match_claim_id"] is True
