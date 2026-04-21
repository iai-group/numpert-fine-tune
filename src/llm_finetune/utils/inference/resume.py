"""Resume-loading logic for inference runs."""

import json
from typing import Tuple

# Fields that every valid inference-details entry must contain.
_REQUIRED_FIELDS = frozenset(
    {
        "index",
        "claim_id",
        "perturbation_type",
        "label",
        "full_prompt",
        "raw_response",
        "parsed_successfully",
        "predicted_label",
    }
)


def _is_valid_entry(entry: object, max_index: int) -> bool:
    """Return True only when *entry* satisfies the full required schema.

    Args:
        entry: Decoded JSON object from one JSONL line.
        max_index: Exclusive upper bound for the ``index`` field
            (i.e. ``len(raw_outputs)``).

    Returns:
        ``True`` if the entry is a dict with all required fields and a
        well-formed ``index`` value; ``False`` otherwise.
    """
    if not isinstance(entry, dict):
        return False
    if not _REQUIRED_FIELDS.issubset(entry.keys()):
        return False
    idx = entry.get("index")
    if not isinstance(idx, int):
        return False
    if not (0 <= idx < max_index):
        return False
    return True


def load_resume_data(
    path: str,
    num_examples: int,
) -> Tuple[list, set]:
    """Load previously completed inference results from a JSONL file.

    Validates the full required schema for each entry before accepting it,
    so that downstream code (duplicate filling, metrics building) can safely
    access every field without risking ``KeyError`` or inconsistent outputs.

    Args:
        path: Path to the ``inference_details.jsonl`` file to load from.
        num_examples: Total number of examples in the current run
            (used to validate ``index`` bounds).

    Returns:
        A two-tuple ``(raw_outputs, completed_indices)`` where:

        * ``raw_outputs`` is a list of length *num_examples*, with valid
          entries in their indexed slots and empty dicts elsewhere.
        * ``completed_indices`` is the set of integer indices that were
          successfully loaded.
    """
    raw_outputs: list = [{} for _ in range(num_examples)]
    completed_indices: set = set()

    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_valid_entry(entry, num_examples):
                continue
            idx = entry["index"]
            raw_outputs[idx] = entry
            completed_indices.add(idx)

    return raw_outputs, completed_indices
