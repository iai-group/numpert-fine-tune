"""loader.py.

Unified data-loading entry point for the NumPert fine-tuning pipeline.

Provides :func:`load_split` as a single, reusable function that handles both
the standard (expansion-only) and balanced (via the dataset balancer) loading
paths for any dataset split (train, validation, test).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset

from src.llm_finetune.utils.data_loading.balancer.dataset_balancer import (
    build_balanced_dataset,
    load_files as _load_raw_json_files,
)
from src.llm_finetune.utils.data_loading.conversation import (
    convert_to_conversations,
)
from src.llm_finetune.utils.data_loading.hf_dataset_instructions import (
    load_fact_checking_data,
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

    When ``use_balancer=False`` (default), delegates directly to
    :func:`~hf_dataset_instructions.load_fact_checking_data` — the original
    expansion-only path.

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
