"""loader.py.

Unified data-loading entry point for the NumPert fine-tuning pipeline.

Provides :func:`load_split` as the primary interface for loading any dataset
split (train, validation, test), routing through either the expansion-only
path or the balanced (via the dataset balancer) path.

Also provides :func:`load_fact_checking_data` (the expansion-only engine),
:func:`load_train_and_validation` (convenience wrapper), and
:func:`load_json_files` (raw JSON loading).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset

from src.llm_finetune.utils.data_loading.balancer.dataset_balancer import (
    build_balanced_dataset,
    load_files as _load_raw_json_files,
)
from src.llm_finetune.utils.data_loading.conversation import (
    convert_to_conversations,
)
from src.llm_finetune.utils.data_loading.perturbation_logic import (
    _normalize_label,
    expand_perturbations,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)


def _resolve_file_paths(
    file_configs: Optional[List[FileLoadConfig]],
    json_paths: Optional[List[str]],
) -> list[Union[str, Path]]:
    """Return a flat list of file paths from either configs or raw paths."""
    return [cfg.path for cfg in (file_configs or [])] or list(json_paths or [])


def _build_allowed_types_by_file(
    file_configs: Optional[List[FileLoadConfig]],
) -> dict[str, set[str]] | None:
    """Build a mapping of filename to allowed perturbation type keys.

    Returns ``None`` when no config has ``perturb_types`` set, meaning
    every perturbation type in every file is allowed.

    The returned keys are source filenames (``Path(path).name``), matching
    the convention used by :func:`~balancer.dataset_balancer.load_files`.
    Values are sets of parsed type keys (e.g. ``"type1"``,
    ``"type3_random"``) suitable for
    :func:`~balancer.dataset_balancer.flatten_samples`.
    """
    if not file_configs:
        return None

    if not any(cfg.perturb_types is not None for cfg in file_configs):
        return None

    mapping: dict[str, set[str]] = {}
    for cfg in file_configs:
        if cfg.perturb_types is None:
            continue
        filename = Path(cfg.path).name
        mapping[filename] = {
            t.value.removeprefix("perturb_")
            for t in cfg.perturb_types
            if t != PerturbationType.ORIGINAL_NORMALIZED
        }
    return mapping


def _build_doc_lookup(file_paths: list[Union[str, Path]]) -> dict[str, str]:
    """Map each claim_id to its evidence document from the raw JSON files."""
    doc_by_claim_id: dict[str, str] = {}
    for _, records in _load_raw_json_files(file_paths):
        for record in records:
            cid = record.get("claim_id")
            if cid and cid not in doc_by_claim_id:
                doc_by_claim_id[cid] = record.get("doc", "")
    return doc_by_claim_id


def _enrich_samples(
    balanced_samples: list[dict],
    doc_by_claim_id: dict[str, str],
) -> list[dict]:
    """Attach doc text and a ``perturbation_type`` string to each sample.

    The batch sampler distinguishes original vs. perturbed examples via
    :attr:`PerturbationType.ORIGINAL_NORMALIZED`.  Arrow characters in type
    strings are replaced with ``"2"`` for ASCII safety (e.g. ``"T→T"`` →
    ``"T2T"``).
    """
    processed: list[dict] = []
    for sample in balanced_samples:
        cid = sample["source_claim_id"]
        perturbation_type = (
            PerturbationType.ORIGINAL_NORMALIZED.value
            if not sample["is_perturbed"]
            else sample["type"].replace("→", "2")
        )
        processed.append(
            {
                "claim_id": cid,
                "claim": sample["text"],
                "doc": doc_by_claim_id.get(cid, ""),
                "label": sample["label"],
                "perturbation_type": perturbation_type,
                "source_file": sample["source_file"],
                "perturbation_method": sample["perturbation_method"] or "",
            }
        )
    return processed


def load_split(
    file_configs: Optional[List[FileLoadConfig]] = None,
    json_paths: Optional[List[str]] = None,
    split: str = "train",
    expand_perturbations: bool = True,
    include_doc: bool = True,
    # Balancing options — all ignored when use_balancer=False
    use_balancer: bool = False,
    balance_labels: bool = False,
    perturb_ratio: float = 1.0,
    false_perturb_t_to_f_ratio: Optional[float] = 0.5,
    equal_perturbation_methods: bool = False,
    seed: int = 3407,
) -> Dataset:
    """Load a dataset split, optionally applying dataset balancing.

    When ``use_balancer=False`` (default), runs the expansion-only path
    via :func:`load_fact_checking_data`.

    When ``use_balancer=True``, loads raw JSON files, applies label/ratio/
    method balancing via
    :func:`~balancer.dataset_balancer.build_balanced_dataset`,
    then converts the result to a HuggingFace Dataset in
    conversation format.
    The ``file_configs`` / ``json_paths`` are mutually
    exclusive; ``file_configs``
    takes precedence when both are provided.

    Args:
        file_configs: Per-file load configs (path + optional type filter).
        json_paths: Simple list of JSON file paths (used when no
            ``file_configs`` given).
        split: Dataset split label — ``"train"``, ``"validation"``, or
            ``"test"``. Passed through to the conversation formatter.
        expand_perturbations: Whether to expand perturbation keys into
            separate examples (ignored when ``use_balancer=True``, where
            expansion is always performed by the balancer).
        include_doc: Whether to include evidence documents in prompts.
        use_balancer: Route through the dataset balancer rather than the
            vanilla expansion pipeline.
        balance_labels: Enforce equal True/False label counts.
        perturb_ratio: Target ratio of perturbed to original samples.
        false_perturb_t_to_f_ratio: Desired share of False-producing
            perturbations that should be T→F; ``None`` disables constraint.
        equal_perturbation_methods: Undersample so each perturbation method
            contributes equally.
        seed: Random seed for deterministic balancing.

    Returns:
        HuggingFace :class:`~datasets.Dataset` ready for SFTTrainer.
    """
    if not use_balancer:
        return load_fact_checking_data(
            json_paths=json_paths,
            file_configs=file_configs,
            split=split,
            expand_perturbations_flag=expand_perturbations,
            include_doc=include_doc,
        )

    # ---- Balanced path -------------------------------------------------- #
    file_paths = _resolve_file_paths(file_configs, json_paths)
    if not file_paths:
        raise ValueError(
            "load_split: at least one file path must be provided "
            "when use_balancer=True"
        )

    doc_by_claim_id = _build_doc_lookup(file_paths)

    allowed_types = _build_allowed_types_by_file(file_configs)

    balanced_samples = build_balanced_dataset(
        file_paths=file_paths,
        balance_labels=balance_labels,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        equal_method_size=equal_perturbation_methods,
        seed=seed,
        allowed_types_by_file=allowed_types,
    )

    processed = _enrich_samples(balanced_samples, doc_by_claim_id)

    conversation_data = convert_to_conversations(
        processed, split=split, include_doc=include_doc
    )
    return Dataset.from_list(conversation_data)


# ----------------------------------------------------------------------- #
# JSON loading
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
# Expansion-only loading (non-balanced path)
# ----------------------------------------------------------------------- #


def load_fact_checking_data(  # noqa: C901
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
        # Derive a short source-file label from the config path
        source_file = Path(config.path).name

        if expand_perturbations_flag:
            processed = expand_perturbations(
                raw_examples, allowed_types=config.perturb_types
            )
        else:
            # Original claims only (normalized) — no expansion
            processed = []
            for example in raw_examples:
                label_str = _normalize_label(example["label"])
                method = example.get("perturbation_method", "")
                processed.append(
                    {
                        "claim_id": example["claim_id"],
                        "claim": example["normalized_claim"],
                        "doc": example["doc"],
                        "label": label_str,
                        "perturbation_type": (
                            PerturbationType.ORIGINAL_NORMALIZED.value
                        ),
                        "url": example.get("url", ""),
                        "taxonomy_label": example.get("taxonomy_label", ""),
                        "perturbation_method": method,
                    }
                )

        # Annotate every expanded row with its source file
        for item in processed:
            item["source_file"] = source_file

        all_processed.extend(processed)

    # Deduplicate originals — when loading from multiple source files,
    # each file produces an original_normalized entry for every claim it
    # contains.  Keep only the first original per claim_id to avoid
    # inflating original counts in the balancer.
    seen_original_ids: set = set()
    deduped: List[Dict] = []
    n_dupes_removed = 0
    for item in all_processed:
        if (
            item["perturbation_type"]
            == PerturbationType.ORIGINAL_NORMALIZED.value
        ):
            if item["claim_id"] in seen_original_ids:
                n_dupes_removed += 1
                continue
            seen_original_ids.add(item["claim_id"])
        deduped.append(item)

    if n_dupes_removed > 0:
        print(
            f"  Deduplicated {n_dupes_removed} duplicate original entries"
            f" (kept {len(seen_original_ids)} unique originals)"
        )
    all_processed = deduped

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
