"""Tests for the inference_config_export utility."""

import json
import pytest
from datetime import datetime

from datasets import Dataset

from src.llm_finetune.utils.inference.inference_config_export import (
    INFERENCE_CONFIG_FILENAME,
    _MAX_PARENT_SEARCH_DEPTH,
    build_inference_config,
    find_training_config,
    save_inference_config,
)
from src.llm_finetune.utils.training.training_config_export import (
    CONFIG_FILENAME as TRAINING_CONFIG_FILENAME,
    build_training_config,
    save_training_config,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

FIXED_TIMESTAMP = datetime(2025, 7, 20, 14, 30, 0)


@pytest.fixture
def sample_test_configs():
    """Return a list of FileLoadConfig for test files."""
    return [
        FileLoadConfig(
            path="data/perturbed_data/test_split/approximation_True.json",
            perturb_types=[
                PerturbationType.PERTURB_TYPE1,
                PerturbationType.PERTURB_TYPE2,
            ],
        ),
        FileLoadConfig(
            path="data/perturbed_data/test_split/numeration_False.json",
            perturb_types=None,
        ),
    ]


@pytest.fixture
def training_config_dir(tmp_path):
    """Create a mock model directory tree with a training_config.json.

    Structure mirrors the fine-tuning output:
        base_dir/
        ├── training_config.json
        └── epoch-3/
            └── merged_model_r32/
    """
    base_dir = tmp_path / "models" / "Qwen3-8B_20250615_120000_type1"
    model_dir = base_dir / "epoch-3" / "merged_model_r32"
    model_dir.mkdir(parents=True)

    training_cfg = build_training_config(
        model_name="unsloth/Qwen3-8B",
        rank=32,
        num_train_epochs=3,
        batch_size=4,
        timestamp=datetime(2025, 6, 15, 12, 0, 0),
    )
    save_training_config(training_cfg, base_dir)

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "training_config": training_cfg,
    }


def _make_dataset(
    n: int = 10,
    perturb_types=None,
    labels=None,
) -> Dataset:
    """Create a minimal HuggingFace dataset for testing."""
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
# find_training_config
# ------------------------------------------------------------------


class TestFindTrainingConfig:
    """Tests for find_training_config."""

    def test_finds_config_in_parent(self, training_config_dir):
        """Discovers training_config.json two levels above model dir."""
        result = find_training_config(str(training_config_dir["model_dir"]))
        assert result is not None
        assert result["model"]["name"] == "unsloth/Qwen3-8B"

    def test_finds_config_in_same_directory(self, tmp_path):
        """Finds training_config.json in the model directory itself."""
        cfg = {"model": {"name": "test"}, "training": {"epochs": 1}}
        config_file = tmp_path / TRAINING_CONFIG_FILENAME
        config_file.write_text(json.dumps(cfg), encoding="utf-8")

        result = find_training_config(str(tmp_path))
        assert result == cfg

    def test_returns_none_for_nonexistent_path(self):
        """Returns None when the model path does not exist."""
        result = find_training_config("/nonexistent/model/path")
        assert result is None

    def test_returns_none_when_no_config_found(self, tmp_path):
        """Returns None when no training_config.json exists anywhere."""
        deep_dir = tmp_path
        for i in range(_MAX_PARENT_SEARCH_DEPTH + 2):
            deep_dir = deep_dir / f"level_{i}"
        deep_dir.mkdir(parents=True)

        result = find_training_config(str(deep_dir))
        assert result is None

    def test_returns_none_for_huggingface_model_name(self):
        """Returns None for a HuggingFace model identifier (not a dir)."""
        result = find_training_config("unsloth/Qwen3-8B")
        assert result is None

    def test_respects_max_depth(self, tmp_path):
        """Does not search beyond _MAX_PARENT_SEARCH_DEPTH levels."""
        # Place config at root
        cfg = {"model": {"name": "too-far"}}
        (tmp_path / TRAINING_CONFIG_FILENAME).write_text(
            json.dumps(cfg), encoding="utf-8"
        )

        # Create a directory deeper than the search depth
        deep_dir = tmp_path
        for i in range(_MAX_PARENT_SEARCH_DEPTH + 1):
            deep_dir = deep_dir / f"level_{i}"
        deep_dir.mkdir(parents=True)

        result = find_training_config(str(deep_dir))
        assert result is None


# ------------------------------------------------------------------
# build_inference_config
# ------------------------------------------------------------------


class TestBuildInferenceConfig:
    """Tests for build_inference_config."""

    def test_minimal_config(self):
        """Builds a valid config with only required arguments."""
        config = build_inference_config(
            model_name="unsloth/Qwen3-8B",
            model_path="unsloth/Qwen3-8B",
            timestamp=FIXED_TIMESTAMP,
        )

        assert config["model"]["name"] == "unsloth/Qwen3-8B"
        assert config["model"]["path"] == "unsloth/Qwen3-8B"
        assert config["timestamp"] == FIXED_TIMESTAMP.isoformat()
        assert config["data"]["test_files"] == []
        assert "test_stats" not in config["data"]
        assert config["training_provenance"] is None

    def test_full_config_with_test_data(self, sample_test_configs):
        """All parameters are recorded in the output dictionary."""
        test_ds = _make_dataset(n=15)

        config = build_inference_config(
            model_name="Qwen3-32B",
            model_path="/models/Qwen3-32B/epoch-3/merged_model_r64",
            test_file_configs=sample_test_configs,
            test_dataset=test_ds,
            rank=64,
            start_epoch=3,
            expand_perturbations=True,
            max_seq_length=8192,
            timestamp=FIXED_TIMESTAMP,
        )

        # Model section
        assert config["model"]["name"] == "Qwen3-32B"
        assert config["model"]["lora_rank"] == 64
        assert config["model"]["start_epoch"] == 3

        # Inference section
        assert config["inference"]["max_seq_length"] == 8192

        # Data section
        d = config["data"]
        assert len(d["test_files"]) == 2
        assert d["test_files"][0]["perturb_types"] == [
            "perturb_type1",
            "perturb_type2",
        ]
        assert d["test_files"][1]["perturb_types"] is None
        assert d["expand_perturbations"] is True
        assert d["test_stats"]["size"] == 15

    def test_includes_training_provenance(self, training_config_dir):
        """Embeds training_config.json when model is a local directory."""
        config = build_inference_config(
            model_name="Qwen3-8B",
            model_path=str(training_config_dir["model_dir"]),
            timestamp=FIXED_TIMESTAMP,
        )

        prov = config["training_provenance"]
        assert prov is not None
        assert prov["model"]["name"] == "unsloth/Qwen3-8B"
        assert prov["model"]["lora_rank"] == 32

    def test_training_provenance_none_for_hub_model(self):
        """training_provenance is None for HuggingFace Hub models."""
        config = build_inference_config(
            model_name="unsloth/Qwen3-8B",
            model_path="unsloth/Qwen3-8B",
            timestamp=FIXED_TIMESTAMP,
        )
        assert config["training_provenance"] is None

    def test_default_timestamp_is_generated(self):
        """When no timestamp is given, a current one is generated."""
        config = build_inference_config(
            model_name="test",
            model_path="test",
        )
        parsed = datetime.fromisoformat(config["timestamp"])
        assert isinstance(parsed, datetime)

    def test_config_is_json_serializable(self, sample_test_configs):
        """The returned dict can be serialized to JSON without errors."""
        config = build_inference_config(
            model_name="test",
            model_path="/some/path",
            test_file_configs=sample_test_configs,
            test_dataset=_make_dataset(n=4),
            timestamp=FIXED_TIMESTAMP,
        )
        serialized = json.dumps(config)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped == config


# ------------------------------------------------------------------
# save_inference_config
# ------------------------------------------------------------------


class TestSaveInferenceConfig:
    """Tests for save_inference_config."""

    def test_creates_file(self, tmp_path):
        """JSON file is created at the expected location."""
        config = build_inference_config(
            model_name="test",
            model_path="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_inference_config(config, tmp_path)

        assert result_path.exists()
        assert result_path.name == INFERENCE_CONFIG_FILENAME
        assert result_path.parent == tmp_path

    def test_file_content_matches(self, tmp_path):
        """Written JSON matches the original dictionary."""
        config = build_inference_config(
            model_name="unsloth/Qwen3-8B",
            model_path="/models/test",
            rank=16,
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_inference_config(config, tmp_path)

        with open(result_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == config

    def test_creates_nested_directories(self, tmp_path):
        """Missing parent directories are created automatically."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        config = build_inference_config(
            model_name="test",
            model_path="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_inference_config(config, nested_dir)

        assert result_path.exists()
        assert result_path.parent == nested_dir

    def test_custom_filename(self, tmp_path):
        """A custom filename can be specified."""
        config = build_inference_config(
            model_name="test",
            model_path="test",
            timestamp=FIXED_TIMESTAMP,
        )
        result_path = save_inference_config(
            config, tmp_path, filename="my_inference.json"
        )

        assert result_path.name == "my_inference.json"
        assert result_path.exists()

    def test_full_roundtrip_with_provenance(
        self, tmp_path, training_config_dir, sample_test_configs
    ):
        """Full roundtrip: build -> save -> load with training provenance."""
        test_ds = _make_dataset(n=8)

        config = build_inference_config(
            model_name="Qwen3-8B",
            model_path=str(training_config_dir["model_dir"]),
            test_file_configs=sample_test_configs,
            test_dataset=test_ds,
            rank=32,
            start_epoch=3,
            max_seq_length=8192,
            timestamp=FIXED_TIMESTAMP,
        )

        result_path = save_inference_config(config, tmp_path)

        with open(result_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Verify inference params
        assert loaded["model"]["name"] == "Qwen3-8B"
        assert loaded["model"]["start_epoch"] == 3
        assert loaded["inference"]["max_seq_length"] == 8192
        assert len(loaded["data"]["test_files"]) == 2
        assert loaded["data"]["test_stats"]["size"] == 8

        # Verify embedded training provenance
        prov = loaded["training_provenance"]
        assert prov is not None
        assert prov["model"]["name"] == "unsloth/Qwen3-8B"
        assert prov["training"]["num_train_epochs"] == 3
