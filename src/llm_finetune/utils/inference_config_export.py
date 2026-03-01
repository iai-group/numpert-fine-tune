"""Export inference configuration and model provenance to a JSON file.

Creates a comprehensive record of the inference run: model identity,
test data files, perturbation types, inference parameters, and — when
available — the training configuration used to produce the model.  The
output JSON is saved alongside the prediction results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.llm_finetune.data_loading_utils.perturbation_types import (
    FileLoadConfig,
)
from src.llm_finetune.utils_training.training_config_export import (
    CONFIG_FILENAME as TRAINING_CONFIG_FILENAME,
    _dataset_stats,
    _file_config_to_dict,
)

INFERENCE_CONFIG_FILENAME = "inference_config.json"

# Maximum number of parent directories to search for training_config.json
_MAX_PARENT_SEARCH_DEPTH = 5


# ------------------------------------------------------------------
# Training config discovery
# ------------------------------------------------------------------


def find_training_config(model_path: str) -> Optional[Dict[str, Any]]:
    """Locate and load the ``training_config.json`` for a local model.

    Walks up the directory tree from *model_path* (up to
    ``_MAX_PARENT_SEARCH_DEPTH`` levels) looking for the training
    provenance file written during fine-tuning.

    Args:
        model_path: Path to the model directory (e.g. the merged model
            or LoRA weights folder).

    Returns:
        The parsed training configuration dictionary, or ``None`` if
        the file was not found or *model_path* is not a local directory.
    """
    path = Path(model_path)
    if not path.is_dir():
        return None

    current = path
    for _ in range(_MAX_PARENT_SEARCH_DEPTH):
        candidate = current / TRAINING_CONFIG_FILENAME
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        if current.parent == current:
            break
        current = current.parent

    return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_inference_config(
    *,
    model_name: str,
    model_path: str,
    test_file_configs: Optional[List[FileLoadConfig]] = None,
    test_dataset=None,
    rank: int = 32,
    start_epoch: int = 0,
    use_vllm: bool = False,
    expand_perturbations: bool = True,
    max_seq_length: int = 16384,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a dictionary describing the full inference configuration.

    Args:
        model_name: Model name as provided by the user.
        model_path: Resolved path or HuggingFace identifier actually
            used to load the model.
        test_file_configs: Per-file load configs for test data.
        test_dataset: The loaded test dataset (for distribution stats).
            ``None`` omits dataset statistics.
        rank: LoRA rank.
        start_epoch: Epoch of the trained model checkpoint.
        use_vllm: Whether vLLM backend was used.
        expand_perturbations: Whether perturbation expansion is enabled.
        max_seq_length: Maximum sequence length in tokens.
        timestamp: Run timestamp (defaults to ``datetime.now()``).

    Returns:
        A JSON-serialisable dictionary with the full inference
        configuration including discovered training provenance.
    """
    if timestamp is None:
        timestamp = datetime.now()

    config: Dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "model": {
            "name": model_name,
            "path": model_path,
            "lora_rank": rank,
            "start_epoch": start_epoch,
        },
        "inference": {
            "use_vllm": use_vllm,
            "max_seq_length": max_seq_length,
        },
        "data": {
            "expand_perturbations": expand_perturbations,
            "test_files": (
                [_file_config_to_dict(c) for c in test_file_configs]
                if test_file_configs
                else []
            ),
        },
    }

    # Test dataset statistics
    if test_dataset is not None:
        config["data"]["test_stats"] = _dataset_stats(test_dataset)

    # Training provenance (from training_config.json if available)
    training_config = find_training_config(model_path)
    if training_config is not None:
        config["training_provenance"] = training_config
    else:
        config["training_provenance"] = None

    return config


def save_inference_config(
    config: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = INFERENCE_CONFIG_FILENAME,
) -> Path:
    """Write an inference configuration dictionary to a JSON file.

    Creates ``output_dir`` if it does not exist.

    Args:
        config: The configuration dictionary (from
            :func:`build_inference_config`).
        output_dir: Directory to write the file into.
        filename: Name of the output file.

    Returns:
        The :class:`Path` to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return output_path
