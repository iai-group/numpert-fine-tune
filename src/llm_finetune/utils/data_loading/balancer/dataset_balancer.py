"""Dataset balancer for claim verification training data.

Loads multiple input JSON files and produces a balanced, deduplicated, and
structured dataset with configurable label balancing, perturbation ratio
constraints, method-level fairness, and cross-file fairness.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PERTURB_FIELD_PATTERN = re.compile(
    r"^perturb_(type\d+(?:_[a-z]+)?)_(True|False)$"
)

EXCLUDED_METHODS = frozenset({"removed"})


@dataclass
class Sample:
    """A single training sample."""

    text: str
    label: str
    source_claim_id: str
    type: str  # "Original", "T→T", "T→F", "F→F"
    perturbation_method: Optional[str]
    is_perturbed: bool
    source_file: str


def load_files(
    file_paths: list[str | Path],
) -> list[tuple[str, list[dict]]]:
    """Load JSON files and return (filename, records) pairs."""
    results = []
    for path in file_paths:
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        results.append((p.name, data))
    return results


def _parse_perturb_key(key: str) -> tuple[str, str] | None:
    """Parse a perturbation field name.

    Returns (perturb_type, original_label_from_key) or None.

    Example:
        "perturb_type3_random_True" -> ("type3_random", "True")
        "perturb_type1_True" -> ("type1", "True")
    """
    match = PERTURB_FIELD_PATTERN.match(key)
    if not match:
        return None
    return match.group(1), match.group(2)


def _determine_perturbed_label(original_label: str, perturb_type: str) -> str:
    """Determine the label of a perturbed sample."""
    if perturb_type == "type1":
        return original_label
    elif perturb_type == "type2":
        return "False"
    elif perturb_type.startswith("type3"):
        return "False"
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")


def _assign_type(original_label: str, perturbed_label: str) -> str:
    """Assign sample type based on original and perturbed labels."""
    if original_label == "True" and perturbed_label == "True":
        return "T→T"
    elif original_label == "True" and perturbed_label == "False":
        return "T→F"
    elif original_label == "False" and perturbed_label == "False":
        return "F→F"
    else:
        raise ValueError(
            f"Unexpected label transition: "
            f"{original_label} → {perturbed_label}"
        )


def flatten_samples(
    file_data: list[tuple[str, list[dict]]],
) -> list[Sample]:
    """Flatten input records into individual samples."""
    samples = []
    for source_file, records in file_data:
        for record in records:
            claim_id = record.get("claim_id")
            original_label = record.get("label")
            normalized_claim = record.get("normalized_claim")
            perturbation_method = record.get("pertubation_method")

            if not claim_id or not original_label or not normalized_claim:
                logger.warning(
                    "Skipping record missing required fields in %s",
                    source_file,
                )
                continue

            # Skip records whose method is excluded entirely
            if perturbation_method in EXCLUDED_METHODS:
                continue

            # Original sample
            samples.append(
                Sample(
                    text=normalized_claim,
                    label=original_label,
                    source_claim_id=claim_id,
                    type="Original",
                    perturbation_method=None,
                    is_perturbed=False,
                    source_file=source_file,
                )
            )

            # Perturbation samples
            for key, value in record.items():
                if not key.startswith("perturb_type"):
                    continue

                parsed = _parse_perturb_key(key)
                if parsed is None:
                    logger.warning("Could not parse perturb key: %s", key)
                    continue

                perturb_type, label_from_key = parsed

                # Skip excluded perturbation sub-types
                # (e.g. type3_removed)
                sub_type = perturb_type.split("_", 1)
                if len(sub_type) > 1 and sub_type[1] in EXCLUDED_METHODS:
                    continue

                # Skip empty values
                if not value or not isinstance(value, str) or not value.strip():
                    continue

                # Validate label consistency
                if label_from_key != original_label:
                    logger.warning(
                        "Label mismatch for %s in claim %s: "
                        "key says %s, record says %s. Skipping.",
                        key,
                        claim_id,
                        label_from_key,
                        original_label,
                    )
                    continue

                perturbed_label = _determine_perturbed_label(
                    original_label, perturb_type
                )
                sample_type = _assign_type(original_label, perturbed_label)

                samples.append(
                    Sample(
                        text=value.strip(),
                        label=perturbed_label,
                        source_claim_id=claim_id,
                        type=sample_type,
                        perturbation_method=perturbation_method,
                        is_perturbed=True,
                        source_file=source_file,
                    )
                )

    return samples


def deduplicate(samples: list[Sample]) -> list[Sample]:
    """Remove samples with duplicate text.

    Keeps first occurrence. Coupling rule is enforced separately.
    """
    seen_texts: set[str] = set()
    result = []
    for sample in samples:
        if sample.text not in seen_texts:
            seen_texts.add(sample.text)
            result.append(sample)
        else:
            logger.debug(
                "Dedup: removed duplicate text from claim %s",
                sample.source_claim_id,
            )
    return result


def _undersample_fair(
    items: list[Sample],
    target_count: int,
    rng: random.Random,
) -> list[Sample]:
    """Undersample to target_count, distributing fairly across files.

    Uses round-robin sampling across source files to avoid exhausting any single
    file's contribution early.
    """
    if len(items) <= target_count:
        return list(items)

    by_file: dict[str, list[Sample]] = defaultdict(list)
    for s in items:
        by_file[s.source_file].append(s)

    # Shuffle within each file for randomness
    for file_samples in by_file.values():
        rng.shuffle(file_samples)

    # Round-robin sample across files
    selected: list[Sample] = []
    file_keys = sorted(by_file.keys())
    file_iters = {k: iter(by_file[k]) for k in file_keys}

    while len(selected) < target_count and file_iters:
        exhausted = []
        for key in list(file_iters.keys()):
            if len(selected) >= target_count:
                break
            try:
                selected.append(next(file_iters[key]))
            except StopIteration:
                exhausted.append(key)
        for key in exhausted:
            del file_iters[key]

    return selected


def _balance_methods(
    perturbed: list[Sample], rng: random.Random
) -> list[Sample]:
    """Balance perturbation methods within each label group."""
    result = []

    for label in ("True", "False"):
        label_samples = [s for s in perturbed if s.label == label]

        by_method: dict[Optional[str], list[Sample]] = defaultdict(list)
        for s in label_samples:
            by_method[s.perturbation_method].append(s)

        if not by_method:
            continue

        min_size = min(len(v) for v in by_method.values())

        for method_samples in by_method.values():
            result.extend(_undersample_fair(method_samples, min_size, rng))

    return result


def _compute_false_perturbation_targets(
    t_to_f_count: int,
    f_to_f_count: int,
    t_to_f_fraction: float,
    target_total: Optional[int] = None,
) -> tuple[int, int]:
    """Compute feasible T→F/F→F counts under a target fraction.

    ``t_to_f_fraction`` is interpreted as the share of False
    perturbations that should be T→F.
    """
    if not 0 <= t_to_f_fraction <= 1:
        raise ValueError("false_perturb_t_to_f_ratio must be between 0 and 1")

    if t_to_f_count == 0 or f_to_f_count == 0:
        kept_t_to_f = t_to_f_count
        kept_f_to_f = f_to_f_count
    elif t_to_f_fraction == 0:
        kept_t_to_f = 0
        kept_f_to_f = f_to_f_count
    elif t_to_f_fraction == 1:
        kept_t_to_f = t_to_f_count
        kept_f_to_f = 0
    else:
        actual_fraction = t_to_f_count / (t_to_f_count + f_to_f_count)

        if actual_fraction > t_to_f_fraction:
            kept_f_to_f = f_to_f_count
            kept_t_to_f = min(
                t_to_f_count,
                int(f_to_f_count * t_to_f_fraction / (1 - t_to_f_fraction)),
            )
        elif actual_fraction < t_to_f_fraction:
            kept_t_to_f = t_to_f_count
            kept_f_to_f = min(
                f_to_f_count,
                int(t_to_f_count * (1 - t_to_f_fraction) / t_to_f_fraction),
            )
        else:
            kept_t_to_f = t_to_f_count
            kept_f_to_f = f_to_f_count

    kept_total = kept_t_to_f + kept_f_to_f
    if target_total is None or kept_total <= target_total:
        return kept_t_to_f, kept_f_to_f

    if target_total <= 0:
        return 0, 0

    if t_to_f_fraction == 0:
        return 0, min(kept_f_to_f, target_total)

    if t_to_f_fraction == 1:
        return min(kept_t_to_f, target_total), 0

    min_t_to_f = max(0, target_total - kept_f_to_f)
    max_t_to_f = min(kept_t_to_f, target_total)
    ideal_t_to_f = target_total * t_to_f_fraction
    candidates = {
        min_t_to_f,
        max_t_to_f,
        max(min_t_to_f, min(max_t_to_f, int(ideal_t_to_f))),
        max(
            min_t_to_f,
            min(max_t_to_f, int(ideal_t_to_f) + 1),
        ),
        max(
            min_t_to_f,
            min(max_t_to_f, round(ideal_t_to_f)),
        ),
    }

    best_t_to_f = min(
        candidates,
        key=lambda count: (
            abs((count / target_total) - t_to_f_fraction),
            abs(count - ideal_t_to_f),
        ),
    )
    return best_t_to_f, target_total - best_t_to_f


def apply_balancing(
    samples: list[Sample],
    balance_labels: bool,
    perturb_ratio: float,
    false_perturb_t_to_f_ratio: Optional[float],
    equal_method_size: bool,
    rng: random.Random,
) -> list[Sample]:
    """Apply all balancing constraints via undersampling.

    Priority (highest first):
        1. Deduplication (already done before this call)
        2. Coupling (done after this call)
        3. Label balancing
        4. Method balancing
        5. Ratio constraints

    We apply from lowest to highest priority so that higher-priority
    constraints get the final say.
    """
    originals = [s for s in samples if not s.is_perturbed]
    perturbed = [s for s in samples if s.is_perturbed]

    # --- 5. Ratio constraints (lowest priority) ---

    true_perturbed = [s for s in perturbed if s.label == "True"]
    false_perturbed = [s for s in perturbed if s.label == "False"]

    t_to_f = [s for s in false_perturbed if s.type == "T→F"]
    f_to_f = [s for s in false_perturbed if s.type == "F→F"]

    target_perturbed = None
    remaining_false_budget = None
    selected_true_perturbed = list(true_perturbed)

    # 5b. Perturbation ratio (perturbed / originals)
    if originals and perturbed:
        target_perturbed = int(len(originals) * perturb_ratio)
        true_budget = min(len(true_perturbed), target_perturbed)
        selected_true_perturbed = _undersample_fair(
            true_perturbed, true_budget, rng
        )
        remaining_false_budget = max(
            0, target_perturbed - len(selected_true_perturbed)
        )

    # 5a. False perturbation composition (share of T→F among
    # False perturbations)
    if false_perturb_t_to_f_ratio is not None:
        target_t_to_f, target_f_to_f = _compute_false_perturbation_targets(
            len(t_to_f),
            len(f_to_f),
            false_perturb_t_to_f_ratio,
            target_total=remaining_false_budget,
        )
        t_to_f = _undersample_fair(t_to_f, target_t_to_f, rng)
        f_to_f = _undersample_fair(f_to_f, target_f_to_f, rng)
        false_perturbed = t_to_f + f_to_f
    elif remaining_false_budget is not None:
        false_perturbed = _undersample_fair(
            false_perturbed, remaining_false_budget, rng
        )

    perturbed = selected_true_perturbed + false_perturbed

    # --- 4. Method balancing ---
    if equal_method_size:
        perturbed = _balance_methods(perturbed, rng)

    # Recombine
    all_samples = originals + perturbed

    # --- 3. Label balancing (highest applied priority) ---
    if balance_labels:
        true_samples = [s for s in all_samples if s.label == "True"]
        false_samples = [s for s in all_samples if s.label == "False"]

        min_count = min(len(true_samples), len(false_samples))

        true_samples = _undersample_fair(true_samples, min_count, rng)
        false_samples = _undersample_fair(false_samples, min_count, rng)

        all_samples = true_samples + false_samples

    return all_samples


def enforce_coupling(
    selected: list[Sample],
    all_samples: list[Sample],
) -> list[Sample]:
    """Ensure originals are included for any selected perturbations.

    If a perturbation from claim_id X is in ``selected``, the
    corresponding original sample must also be present, even if
    it would otherwise be a duplicate.
    """
    selected_perturbed_ids = {
        s.source_claim_id for s in selected if s.is_perturbed
    }

    existing_original_ids = {
        s.source_claim_id for s in selected if not s.is_perturbed
    }

    missing_ids = selected_perturbed_ids - existing_original_ids

    if not missing_ids:
        return selected

    # Find originals from full pool
    originals_by_id: dict[str, Sample] = {}
    for s in all_samples:
        if (
            not s.is_perturbed
            and s.source_claim_id in missing_ids
            and s.source_claim_id not in originals_by_id
        ):
            originals_by_id[s.source_claim_id] = s

    added = list(originals_by_id.values())
    if added:
        logger.info(
            "Coupling: added %d original samples for perturbations.",
            len(added),
        )

    return selected + added


def compute_stats(samples: list[dict]) -> dict:
    """Compute summary statistics for the dataset."""
    label_dist: dict[str, int] = defaultdict(int)
    type_dist: dict[str, int] = defaultdict(int)
    method_dist: dict[str, int] = defaultdict(int)
    file_dist: dict[str, int] = defaultdict(int)

    for s in samples:
        label_dist[s["label"]] += 1
        type_dist[s["type"]] += 1
        if s["perturbation_method"] is not None:
            method_dist[s["perturbation_method"]] += 1
        file_dist[s["source_file"]] += 1

    return {
        "total_samples": len(samples),
        "label_distribution": dict(label_dist),
        "type_distribution": dict(type_dist),
        "method_distribution": dict(method_dist),
        "file_distribution": dict(file_dist),
    }


def build_balanced_dataset(
    file_paths: list[str | Path],
    balance_labels: bool = True,
    perturb_ratio: float = 1.0,
    false_perturb_t_to_f_ratio: float | None = None,
    equal_method_size: bool = False,
    seed: int | None = None,
) -> list[dict]:
    """Build a balanced, deduplicated dataset from multiple JSON files.

    Args:
        file_paths: Paths to JSON files containing claim records.
        balance_labels: Enforce equal True/False sample counts.
        perturb_ratio: Target ratio of perturbed to original samples.
        false_perturb_t_to_f_ratio: Desired share of False
            perturbations that should be T→F. ``0.5`` means equal
            T→F/F→F counts. ``None`` disables the constraint.
        equal_method_size: Balance perturbation methods equally
            within each label group.
        seed: Random seed for deterministic output.

    Returns:
        List of sample dicts with keys: text, label, source_claim_id,
        type, perturbation_method, is_perturbed, source_file.
    """
    rng = random.Random(seed)

    # 1. Load
    file_data = load_files(file_paths)

    # 2. Flatten
    all_samples = flatten_samples(file_data)

    # 3. Deduplicate
    deduped = deduplicate(all_samples)

    # 4. Balance
    balanced = apply_balancing(
        deduped,
        balance_labels=balance_labels,
        perturb_ratio=perturb_ratio,
        false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
        equal_method_size=equal_method_size,
        rng=rng,
    )

    # 5. Coupling
    coupled = enforce_coupling(balanced, deduped)

    # 6. Shuffle
    rng.shuffle(coupled)

    # Convert to dicts
    return [
        {
            "text": s.text,
            "label": s.label,
            "source_claim_id": s.source_claim_id,
            "type": s.type,
            "perturbation_method": s.perturbation_method,
            "is_perturbed": s.is_perturbed,
            "source_file": s.source_file,
        }
        for s in coupled
    ]
