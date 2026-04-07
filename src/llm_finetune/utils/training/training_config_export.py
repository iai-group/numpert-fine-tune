"""Export training configuration details to a JSON file.

Creates a comprehensive record of all parameters, data files, and perturbation
types used during a fine-tuning run.  The output JSON is saved to the model
output directory so that every trained model carries a machine-readable
provenance file.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.llm_finetune.utils.data_loading.perturbation_types import (  # noqa: F401, E501
    FileLoadConfig,
    PerturbationType,
)

CONFIG_FILENAME = "training_config.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _file_config_to_dict(config: FileLoadConfig) -> Dict[str, Any]:
    """Serialize a single :class:`FileLoadConfig` to a plain dict.

    Args:
        config: A file load configuration.

    Returns:
        Dictionary with ``path`` and ``perturb_types`` keys.
    """
    return {
        "path": str(config.path),
        "perturb_types": (
            [t.value for t in config.perturb_types]
            if config.perturb_types is not None
            else None
        ),
    }


def _dataset_stats(dataset) -> Dict[str, Any]:
    """Compute perturbation and label distributions for a dataset.

    Works with any object that exposes a column via indexing
    (HuggingFace :class:`Dataset` or a plain list of dicts).

    Args:
        dataset: A HuggingFace Dataset (or compatible) with
            ``perturbation_type`` and ``label`` columns.

    Returns:
        Dictionary containing size, perturbation distribution, and
        label distribution.
    """
    size = len(dataset)
    if size == 0:
        return {
            "size": 0,
            "perturbation_distribution": {},
            "label_distribution": {},
        }

    columns = dataset.column_names if hasattr(dataset, "column_names") else []
    perturb_dist = (
        dict(Counter(dataset["perturbation_type"]).most_common())
        if "perturbation_type" in columns
        else {}
    )
    label_dist = (
        dict(Counter(dataset["label"]).most_common())
        if "label" in columns
        else {}
    )

    return {
        "size": size,
        "perturbation_distribution": perturb_dist,
        "label_distribution": label_dist,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_training_config(
    *,
    model_name: str,
    train_file_configs: Optional[List[FileLoadConfig]] = None,
    val_file_configs: Optional[List[FileLoadConfig]] = None,
    train_dataset=None,
    val_dataset=None,
    rank: int = 32,
    num_train_epochs: int = 1,
    start_epoch: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 16384,
    learning_rate: float = 1e-4,
    use_perturbations: bool = True,
    balance_labels: bool = False,
    perturb_ratio: Optional[float] = None,
    false_perturb_t_to_f_ratio: Optional[float] = None,
    method_balance_mode: Optional[str] = None,
    batch_match_claim_id: bool = False,
    split_large_claims: bool = True,
    ensure_original_in_batch: bool = True,
    ensure_perturbed_in_batch: bool = True,
    save_gguf: bool = False,
    gguf_quantization_method: str = "q8_0",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a dictionary describing the full training configuration.

    All parameters mirror those of :func:`finetune` in
    ``unsloth_finetune.py``.

    Args:
        model_name: HuggingFace model identifier.
        train_file_configs: Per-file load configs for training data.
        val_file_configs: Per-file load configs for validation data.
        train_dataset: The loaded training dataset (for distribution
            stats).  ``None`` omits dataset statistics.
        val_dataset: The loaded validation dataset.
        rank: LoRA rank.
        num_train_epochs: Number of epochs to train.
        start_epoch: Epoch to resume from.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_seq_length: Maximum sequence length in tokens.
        learning_rate: Optimizer learning rate.
        use_perturbations: Whether perturbation expansion is enabled.
        balance_labels: Whether label balancing is applied.
        perturb_ratio: Ratio of perturbed to original examples.
        false_perturb_t_to_f_ratio: Among *False* perturbed examples,
            the proportion that are True→False flips (e.g. ``0.67``
            means 67 % of False examples come from True→False
            perturbations).  ``None`` disables this ratio constraint
            and keeps the natural distribution of False examples.
        method_balance_mode: How to balance perturbation methods
            (``"undersample"``, ``"oversample"``, ``"none"``, or ``None``).
        batch_match_claim_id: Whether claim-id batching is active.
        split_large_claims: Split large claim groups across batches.
        ensure_original_in_batch: Require an original example per batch.
        ensure_perturbed_in_batch: Require a perturbed example per batch.
        save_gguf: Whether GGUF export is enabled.
        gguf_quantization_method: Quantization method for GGUF.
        timestamp: Run timestamp (defaults to ``datetime.now()``).

    Returns:
        A JSON-serialisable dictionary with the full configuration.
    """
    if timestamp is None:
        timestamp = datetime.now()

    config: Dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "model": {
            "name": model_name,
            "lora_rank": rank,
        },
        "training": {
            "start_epoch": start_epoch,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "learning_rate": learning_rate,
        },
        "data": {
            "use_perturbations": use_perturbations,
            "balance_labels": balance_labels,
            "perturb_ratio": perturb_ratio,
            "false_perturb_t_to_f_ratio": false_perturb_t_to_f_ratio,
            "method_balance_mode": method_balance_mode,
            "train_files": (
                [_file_config_to_dict(c) for c in train_file_configs]
                if train_file_configs
                else []
            ),
            "val_files": (
                [_file_config_to_dict(c) for c in val_file_configs]
                if val_file_configs
                else []
            ),
        },
        "batch_sampler": {
            "batch_match_claim_id": batch_match_claim_id,
            "split_large_claims": split_large_claims,
            "ensure_original_in_batch": ensure_original_in_batch,
            "ensure_perturbed_in_batch": ensure_perturbed_in_batch,
        },
        "gguf": {
            "save_gguf": save_gguf,
            "quantization_method": gguf_quantization_method,
        },
    }

    # Dataset statistics (only when datasets are provided)
    if train_dataset is not None:
        config["data"]["train_stats"] = _dataset_stats(train_dataset)
    if val_dataset is not None:
        config["data"]["val_stats"] = _dataset_stats(val_dataset)

    return config


def save_training_config(
    config: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = CONFIG_FILENAME,
) -> Path:
    """Write a training configuration dictionary to a JSON file.

    Creates ``output_dir`` if it does not exist.

    Args:
        config: The configuration dictionary (from
            :func:`build_training_config`).
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
