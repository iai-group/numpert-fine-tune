"""hf_dataset_instructions.py.

High-level data loading API for the NumPert fine-tuning pipeline.

This module is the **orchestration layer** that wires together:
  - :mod:`perturbation_types` — enum, convenience sets, ``FileLoadConfig``
  - :mod:`perturbation_logic`  — key detection, label logic, expansion
  - :mod:`conversation`        — SFTTrainer conversation formatting

All public symbols from those sub-modules are re-exported here so that
existing imports continue to work unchanged.
"""

from __future__ import annotations

import json
from datasets import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union

# Re-export taxonomy & config -------------------------------------------
from src.llm_finetune.data_loading_utils.perturbation_types import (  # noqa: F401
    PerturbationType,
    STRUCTURE_PRESERVING_TYPES,
    VALUE_MODIFYING_TYPES,
    OBFUSCATION_TYPES,
    ALL_PERTURB_TYPES,
    FileLoadConfig,
    parse_file_config,
)

# Re-export perturbation logic ------------------------------------------
from src.llm_finetune.data_loading_utils.perturbation_logic import (  # noqa: F401
    _PERTURB_KEY_RE,
    _detect_perturb_key,
    _normalize_label,
    _determine_perturbed_label,
    expand_perturbations,
)

# Re-export conversation formatting -------------------------------------
from src.llm_finetune.data_loading_utils.conversation import (  # noqa: F401
    convert_to_conversations,
)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def load_json_files(file_paths: Union[str, List[str]]) -> List[Dict]:
    """Load one or multiple JSON files.

    Args:
        file_paths: Single file path or list of file paths.

    Returns:
        List of all examples from all files.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_examples: List[Dict] = []
    for file_path in file_paths:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_examples.extend(data)
            else:
                all_examples.append(data)

    return all_examples


# ---------------------------------------------------------------------------
# High-level loading API
# ---------------------------------------------------------------------------


def load_fact_checking_data(
    json_paths: Union[str, List[str], None] = None,
    file_configs: Optional[List[FileLoadConfig]] = None,
    split: str = "train",
    expand_perturbations_flag: bool = True,
    include_doc: bool = True,
) -> Dataset:
    """Load fact-checking data and convert to conversation format.

    Supports two calling conventions:

    1. **Simple** (backward-compatible) — pass *json_paths* to load all
       perturbation types from every file::

           load_fact_checking_data(json_paths=["approx_True.json"])

    2. **Per-file control** — pass *file_configs* to choose which
       perturbation types to load from each file::

           load_fact_checking_data(file_configs=[
               FileLoadConfig("approx_True.json",
                   [PerturbationType.PERTURB_TYPE1,
                    PerturbationType.PERTURB_TYPE2]),
               FileLoadConfig("approx_False.json",
                   [PerturbationType.PERTURB_TYPE1]),
           ])

    Args:
        json_paths: Path(s) to JSON files (loads all types from each).
        file_configs: List of :class:`FileLoadConfig` for per-file control.
        split: ``"train"``, ``"validation"``, or ``"test"``.
        expand_perturbations_flag: Whether to expand perturbation keys into
            separate examples.
        include_doc: Whether to include evidence documents in prompts.

    Returns:
        HuggingFace Dataset ready for SFTTrainer.
    """
    if file_configs is None and json_paths is not None:
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        file_configs = [FileLoadConfig(path=p) for p in json_paths]

    if file_configs is None:
        raise ValueError("Either json_paths or file_configs must be provided")

    effective_split = "validation" if split == "test" else split

    all_processed: List[Dict] = []

    for config in file_configs:
        raw_examples = load_json_files(config.path)

        if expand_perturbations_flag:
            processed = expand_perturbations(
                raw_examples, allowed_types=config.perturb_types
            )
        else:
            # Original claims only (normalized) — no expansion
            processed = []
            for example in raw_examples:
                label_str = _normalize_label(example["label"])
                processed.append(
                    {
                        "claim_id": example["claim_id"],
                        "claim": example["normalized_claim"],
                        "doc": example["doc"],
                        "label": label_str,
                        "perturbation_type": PerturbationType.ORIGINAL_NORMALIZED.value,
                        "url": example.get("url", ""),
                        "taxonomy_label": example.get("taxonomy_label", ""),
                    }
                )

        all_processed.extend(processed)

    # Convert to conversation format
    conversation_data = convert_to_conversations(
        all_processed, split=effective_split, include_doc=include_doc
    )

    return Dataset.from_list(conversation_data)


def load_train_and_validation(
    train_json_paths: Union[str, List[str], None] = None,
    val_json_paths: Union[str, List[str], None] = None,
    train_file_configs: Optional[List[FileLoadConfig]] = None,
    val_file_configs: Optional[List[FileLoadConfig]] = None,
    expand_perturbations_flag: bool = True,
    include_doc: bool = True,
) -> Dict[str, Dataset]:
    """Load both training and validation datasets.

    Accepts either simple paths (backward-compatible) or
    :class:`FileLoadConfig` lists for per-file type control.

    Args:
        train_json_paths: Path(s) to training JSON files.
        val_json_paths: Path(s) to validation JSON files.
        train_file_configs: Per-file configs for training data.
        val_file_configs: Per-file configs for validation data.
        expand_perturbations_flag: Whether to expand perturbations.
        include_doc: Whether to include evidence documents.

    Returns:
        Dictionary with ``"train"`` and ``"validation"`` datasets.
    """
    train_dataset = load_fact_checking_data(
        json_paths=train_json_paths,
        file_configs=train_file_configs,
        split="train",
        expand_perturbations_flag=expand_perturbations_flag,
        include_doc=include_doc,
    )

    val_dataset = load_fact_checking_data(
        json_paths=val_json_paths,
        file_configs=val_file_configs,
        split="validation",
        expand_perturbations_flag=expand_perturbations_flag,
        include_doc=include_doc,
    )

    return {"train": train_dataset, "validation": val_dataset}


# ---------------------------------------------------------------------------
# CLI entry point (for quick inspection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.llm_finetune.data_loading_utils"
            ".hf_dataset_instructions <file_config> [file_config ...]"
        )
        print("  file_config format: path/to/file.json[:type1,type2,...]")
        print(
            "  Example: data/perturbed_data/train_split/"
            "approximation_True.json:type1,type2"
        )
        sys.exit(1)

    configs = [parse_file_config(arg) for arg in sys.argv[1:]]
    print(f"Loading {len(configs)} file(s):")
    for c in configs:
        print(f"  {c}")

    dataset = load_fact_checking_data(
        file_configs=configs,
        split="train",
        expand_perturbations_flag=True,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")

    perturb_counts = Counter(dataset["perturbation_type"])
    print("\nPerturbation type distribution:")
    for p_type, count in perturb_counts.most_common():
        print(f"  {p_type:30s}: {count}")

    label_counts = Counter(dataset["label"])
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:30s}: {count}")
