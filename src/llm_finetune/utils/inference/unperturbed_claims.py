"""Extract unique unperturbed (normalized) claims from a loaded dataset.

When multiple perturbation files are provided as input, the same
``claim_id`` appears in each file.  This helper deduplicates so that
each normalized claim is evaluated exactly once.
"""

from __future__ import annotations

from datasets import Dataset


def deduplicate_unperturbed_claims(dataset: Dataset) -> Dataset:
    """Return a dataset with one entry per unique ``claim_id``.

    Keeps the first occurrence of each ``claim_id`` and discards
    subsequent duplicates.  This is intended to be called on a dataset
    that was loaded with ``use_perturbations_flag=False`` (i.e. only
    ``original_normalized`` entries).

    Args:
        dataset: HuggingFace Dataset containing at least a ``claim_id``
            column.

    Returns:
        Filtered dataset with unique ``claim_id`` values.
    """
    if len(dataset) == 0:
        return dataset

    seen: set[str] = set()
    keep_indices: list[int] = []

    for idx, claim_id in enumerate(dataset["claim_id"]):
        if claim_id not in seen:
            seen.add(claim_id)
            keep_indices.append(idx)

    return dataset.select(keep_indices)
