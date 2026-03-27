"""perturbation_logic.py.

Core perturbation expansion engine.

Handles key detection via regex, label normalisation, label-flip logic,
and the main :func:`expand_perturbations` function that turns a raw JSON
example into one-row-per-perturbation-variant.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.llm_finetune.utils.data_loading.perturbation_types import (
    PerturbationType,
)


# ---------------------------------------------------------------------------
# Internal helpers — key detection & label logic
# ---------------------------------------------------------------------------

#: Regex to extract the base perturbation type from a JSON key.
#: Matches keys like ``perturb_type1_True``, ``perturb_type2_False``,
#: ``perturb_type3_random``, ``perturb_type3_masked_False``, etc.
_PERTURB_KEY_RE = re.compile(
    r"^(perturb_type1|perturb_type2"
    r"|perturb_type3_random|perturb_type3_masked|perturb_type3_removed)"
    r"(?:_(?:True|False))?$"
)


def _detect_perturb_key(key: str) -> Optional[str]:
    """Return the base perturbation type for a JSON key, or ``None``.

    >>> _detect_perturb_key("perturb_type1_True")
    'perturb_type1'
    >>> _detect_perturb_key("perturb_type3_random_False")
    'perturb_type3_random'
    >>> _detect_perturb_key("claim_id")  # not a perturbation key
    """
    m = _PERTURB_KEY_RE.match(key)
    return m.group(1) if m else None


def _normalize_label(label) -> str:
    """Normalize a label value (bool or string) to ``"True"``/``"False"``."""
    if isinstance(label, bool):
        return "True" if label else "False"
    if isinstance(label, str):
        lowered = label.strip().lower()
        if lowered == "true":
            return "True"
        if lowered == "false":
            return "False"
    raise ValueError(f"Cannot normalize label: {label!r}")


def _determine_perturbed_label(
    base_type: str, original_label_bool: bool
) -> str:
    """Determine the correct label for a perturbed claim.

    Args:
        base_type: One of the ``PerturbationType`` values
            (e.g. ``"perturb_type1"``).
        original_label_bool: The original claim's truth value.

    Returns:
        ``"True"`` or ``"False"``
    """
    if base_type == PerturbationType.PERTURB_TYPE1.value:
        # Structure-preserving → same label as original
        return "True" if original_label_bool else "False"
    elif base_type == PerturbationType.PERTURB_TYPE2.value:
        # Value-modifying → always False
        return "False"
    else:
        # Type 3 (obfuscation) → always False
        return "False"


# ---------------------------------------------------------------------------
# Perturbation expansion
# ---------------------------------------------------------------------------


def expand_perturbations(
    examples: List[Dict],
    allowed_types: Optional[List[PerturbationType]] = None,
) -> List[Dict]:
    """Expand each example into multiple examples per perturbation key.

    For every example the function always emits an *original* entry (using
    ``normalized_claim``).  It then scans the example's keys for perturbation
    entries, filters them against *allowed_types* (if given), computes the
    correct label, and emits one additional entry per perturbation.

    Args:
        examples: Raw examples loaded from a JSON file.
        allowed_types: If not ``None``, only these perturbation types will
            be included.
            ``PerturbationType.ORIGINAL_NORMALIZED`` is always included
            regardless.

    Returns:
        Expanded list with one entry per perturbation variant plus the
        original.
    """
    # Build the set of allowed base-type strings for fast lookup
    allowed_base_types: Optional[set] = None
    if allowed_types is not None:
        allowed_base_types = {
            t.value
            for t in allowed_types
            if t != PerturbationType.ORIGINAL_NORMALIZED
        }

    expanded: List[Dict] = []

    for example in examples:
        # --- original entry ---
        original_label_str = _normalize_label(example["label"])
        original_label_bool = original_label_str == "True"

        original_entry = {
            "claim_id": example["claim_id"],
            "claim": example["normalized_claim"],
            "doc": example["doc"],
            "label": original_label_str,
            "perturbation_type": PerturbationType.ORIGINAL_NORMALIZED.value,
            "url": example.get("url", ""),
            "taxonomy_label": example.get("taxonomy_label", ""),
        }
        expanded.append(original_entry)

        # --- perturbed entries ---
        for key in example:
            base_type = _detect_perturb_key(key)
            if base_type is None:
                continue  # not a perturbation key

            # Skip empty/null perturbation values
            value = example[key]
            if not value:
                continue

            # Filter by allowed types
            if (
                allowed_base_types is not None
                and base_type not in allowed_base_types
            ):
                continue

            perturbed_label = _determine_perturbed_label(
                base_type, original_label_bool
            )

            perturbed_entry = {
                "claim_id": example["claim_id"],
                "claim": value,
                "doc": example["doc"],
                "label": perturbed_label,
                "perturbation_type": base_type,
                "url": example.get("url", ""),
                "taxonomy_label": example.get("taxonomy_label", ""),
            }
            expanded.append(perturbed_entry)

    return expanded
