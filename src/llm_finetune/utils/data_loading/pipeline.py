"""pipeline.py.

Unified data-loading pipeline for the NumPert fine-tuning project.

This module is the **single entry point** for loading any dataset split.
It always routes through :mod:`dataset_balancer` as the core engine,
using it both for balanced *and* unbalanced loading (the latter simply
skips the balancing constraints).

Public API
----------
:func:`load_split` — primary interface for all callers.

The file-config notation ``"path/to/file.json:type2,type3_masked"``
is supported and parsed by :func:`~perturbation_types.parse_file_config`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset

from src.llm_finetune.utils.data_loading.conversation import (
    convert_to_conversations,
)
from src.llm_finetune.utils.data_loading.dataset_balancer import (
    MethodBalanceMode,
    build_balanced_dataset,
    load_files as _load_raw_json_files,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)
from src.numpert_prompt.prompt import PromptType

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
# Internal helpers
# ----------------------------------------------------------------------- #


def _resolve_file_paths(
    file_configs: Optional[List[FileLoadConfig]],
    json_paths: Optional[Union[str, List[str]]],
) -> list[Union[str, Path]]:
    """Return a flat list of file paths from either configs or raw paths."""
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    return [cfg.path for cfg in (file_configs or [])] or list(json_paths or [])


def _build_allowed_types_by_file(
    file_configs: Optional[List[FileLoadConfig]],
) -> dict[str, set[str]] | None:
    """Build a mapping of filename → allowed perturbation type keys.

    Returns ``None`` when no config has ``perturb_types`` set (= keep all).
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


def _build_doc_lookup(
    file_paths: list[Union[str, Path]],
) -> dict[str, str]:
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

    Arrow characters in type strings are replaced with ``"2"`` for ASCII
    safety (e.g. ``"T→T"`` → ``"T2T"``).
    """
    processed: list[dict] = []
    for sample in balanced_samples:
        cid = sample["source_claim_id"]
        if not sample["is_perturbed"]:
            perturbation_type = PerturbationType.ORIGINAL_NORMALIZED.value
        elif sample.get("perturb_type_key"):
            perturbation_type = f"perturb_{sample['perturb_type_key']}"
        else:
            perturbation_type = sample["type"].replace("→", "2")
        processed.append(
            {
                "claim_id": cid,
                "claim": sample["text"],
                "doc": doc_by_claim_id.get(cid, ""),
                "label": sample["label"],
                "perturbation_type": perturbation_type,
                "sample_type": sample["type"],
                "source_file": sample["source_file"],
                "perturbation_method": sample["perturbation_method"] or "",
            }
        )
    return processed


# ----------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------- #


def load_split(
    file_configs: Optional[List[FileLoadConfig]] = None,
    json_paths: Optional[Union[str, List[str]]] = None,
    split: str = "train",
    use_perturbations: bool = True,
    include_doc: bool = True,
    prompt_type: PromptType = PromptType.ZERO_SHOT,
    # Balancing options
    balance_labels: bool = False,
    perturb_ratio: Optional[float] = None,
    false_perturb_t_to_f_ratio: Optional[float] = None,
    method_balance_mode: MethodBalanceMode | None = None,
    seed: int = 3407,
) -> Dataset:
    """Load a dataset split through a single unified pipeline.

    All data flows through :func:`~dataset_balancer.build_balanced_dataset`,
    which handles loading, expansion, deduplication, and (optionally)
    balancing in one pass.  When no balancing flags are set the balancer
    simply returns the expanded, deduplicated dataset without any
    under-/over-sampling.

    The ``file_configs`` / ``json_paths`` parameters are mutually
    exclusive; ``file_configs`` takes precedence when both are provided.

    Args:
        file_configs: Per-file load configs (path + optional type filter).
            Supports CLI-style notation via
            :func:`~perturbation_types.parse_file_config`, e.g.
            ``"data/perturbed_data/test_split/negative_False.json:type2"``.
        json_paths: Simple list of JSON file paths (used when no
            ``file_configs`` given).
        split: Dataset split label — ``"train"``, ``"validation"``, or
            ``"test"``.
        use_perturbations: Whether to include perturbed examples.
            When ``False``, only original (normalised) claims are loaded.
        include_doc: Whether to include evidence documents in prompts.
        prompt_type: Controls which example block is injected into the user
            prompt. Defaults to ``PromptType.ZERO_SHOT`` (no examples).
        balance_labels: Enforce equal True/False label counts.
        perturb_ratio: Target ratio of perturbed to original samples.
            ``None`` (the default) keeps all perturbations.
        false_perturb_t_to_f_ratio: Desired share of False-producing
            perturbations that should be T→F; ``None`` disables constraint.
        method_balance_mode: How to balance perturbation methods
            (``"undersample"``, ``"oversample"``, ``"none"``, or ``None``).
        seed: Random seed for deterministic output.

    Returns:
        HuggingFace :class:`~datasets.Dataset` ready for SFTTrainer.
    """
    file_paths = _resolve_file_paths(file_configs, json_paths)
    if not file_paths:
        raise ValueError(
            "load_split: at least one file path must be provided "
            "(via file_configs or json_paths)"
        )

    # Build doc lookup from the raw files
    doc_by_claim_id = _build_doc_lookup(file_paths)

    # Build per-file type filtering from file_configs
    allowed_types = _build_allowed_types_by_file(file_configs)

    # When use_perturbations=False we restrict to no perturbation types,
    # so the balancer only produces originals.
    if not use_perturbations:
        # Empty set per file → no perturbed keys will pass the filter
        allowed_types = {Path(p).name: set() for p in file_paths}

    # Route everything through the balancer.  When no balancing is
    # requested the balancer still performs:
    #   load → flatten/expand → deduplicate → coupling → shuffle
    balanced_samples = build_balanced_dataset(
        file_paths=file_paths,
        balance_labels=balance_labels,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        method_balance_mode=method_balance_mode,
        seed=seed,
        allowed_types_by_file=allowed_types,
    )

    logger.info(
        "load_split: %d samples after balancer (split=%s)",
        len(balanced_samples),
        split,
    )

    # Enrich with doc text + perturbation_type labels
    processed = _enrich_samples(balanced_samples, doc_by_claim_id)

    # Convert to conversation format for SFTTrainer
    effective_split = "validation" if split == "test" else split
    conversation_data = convert_to_conversations(
        processed,
        split=effective_split,
        include_doc=include_doc,
        prompt_type=prompt_type,
    )
    return Dataset.from_list(conversation_data)
