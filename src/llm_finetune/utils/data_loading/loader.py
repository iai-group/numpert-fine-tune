"""loader.py.

Backward-compatible data-loading entry point.

.. deprecated::
    This module delegates all work to :mod:`pipeline`, which is the
    **single unified** data-loading engine.  New code should import from
    :mod:`pipeline` directly.  The functions here are thin wrappers kept
    for backward compatibility.

Public symbols re-exported for existing callers:

* :func:`load_split`
* :func:`load_fact_checking_data`
* :func:`load_json_files`
* :func:`load_train_and_validation`
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset

from src.llm_finetune.utils.data_loading.pipeline import (  # noqa: F401
    load_split,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
)


# ----------------------------------------------------------------------- #
# JSON loading — kept here because sequence_len_utility imports it
# ----------------------------------------------------------------------- #


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


# ----------------------------------------------------------------------- #
# Deprecated wrappers
# ----------------------------------------------------------------------- #


def load_fact_checking_data(
    json_paths: Union[str, List[str], None] = None,
    file_configs: Optional[List[FileLoadConfig]] = None,
    split: str = "train",
    expand_perturbations_flag: bool = True,
    include_doc: bool = True,
) -> Dataset:
    """Load fact-checking data and convert to conversation format.

    .. deprecated::
        Use :func:`pipeline.load_split` instead.  This wrapper delegates
        to it.
    """
    warnings.warn(
        "load_fact_checking_data() is deprecated — use "
        "pipeline.load_split() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if file_configs is None and json_paths is not None:
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        file_configs = [FileLoadConfig(path=p) for p in json_paths]

    return load_split(
        file_configs=file_configs,
        json_paths=json_paths,
        split=split,
        expand_perturbations=expand_perturbations_flag,
        include_doc=include_doc,
    )


def load_train_and_validation(
    train_json_paths: Union[str, List[str], None] = None,
    val_json_paths: Union[str, List[str], None] = None,
    train_file_configs: Optional[List[FileLoadConfig]] = None,
    val_file_configs: Optional[List[FileLoadConfig]] = None,
    expand_perturbations_flag: bool = True,
    include_doc: bool = True,
) -> Dict[str, Dataset]:
    """Load both training and validation datasets.

    .. deprecated::
        Use :func:`pipeline.load_split` instead.  This wrapper delegates
        to it.
    """
    warnings.warn(
        "load_train_and_validation() is deprecated — use "
        "pipeline.load_split() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    train_dataset = load_split(
        file_configs=train_file_configs,
        json_paths=train_json_paths,
        split="train",
        expand_perturbations=expand_perturbations_flag,
        include_doc=include_doc,
    )

    val_dataset = load_split(
        file_configs=val_file_configs,
        json_paths=val_json_paths,
        split="validation",
        expand_perturbations=expand_perturbations_flag,
        include_doc=include_doc,
    )

    return {"train": train_dataset, "validation": val_dataset}
