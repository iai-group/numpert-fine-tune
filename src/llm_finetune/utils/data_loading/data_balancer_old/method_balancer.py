"""method_balancer.py.

Balances a HuggingFace Dataset by **perturbation method** (numeration,
range, approximation, masked, random, negative, removed) and **class label**
(True / False).

Two balancing modes
-------------------
1. **Equal-method** (``equal_methods=True``):
   Each perturbation method gets the same number of samples.  Within each
   method the class ratio is governed by ``class_ratio``.

2. **Proportional** (``equal_methods=False``, the default):
   Methods keep their natural proportions, but the total is capped at
   ``max_samples`` (or the full dataset if ``None``).  A minimum-share
   guarantee ensures no method is arbitrarily excluded: every method
   receives at least ``floor(budget / n_methods)`` samples (capped by its
   actual size).

Both modes respect an overall ``class_ratio`` (True : False).  Set to
``None`` to skip class balancing entirely.

Future option (not yet implemented)
-----------------------------------
``match_claim_ids``: when set, ensures the *same* claim_id is selected
across all perturbation methods so each claim's full set of variations
is present.  Tracked as a TODO.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import Dataset

from src.llm_finetune.utils.data_loading.perturbation_types import (
    PerturbationType,
)

# Known perturbation methods (for reference / logging).
KNOWN_METHODS = frozenset(
    {
        "approximation",
        "numeration",
        "range",
        "masked",
        "random",
        "negative",
        "removed",
    }
)


class MethodBalancer:
    """Balance a dataset by perturbation method and class label.

    Args:
        dataset: HuggingFace Dataset with at least ``perturbation_method``,
            ``perturbation_type``, ``claim_id``, and ``label`` columns
            (label may also be extracted from ``conversations``).
        equal_methods: If True, equalise the sample count across methods.
        class_ratio: Desired True:False ratio (e.g. 1.0 = 50/50).
            ``None`` disables class balancing.
        max_samples: Hard cap on total output size.
            ``None`` means no cap (use all available, after method /
            class adjustments).
        include_originals: Whether to always include all ORIGINAL_NORMALIZED
            examples.  When True the originals are included as-is and do
            *not* count toward the per-method budgets.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        equal_methods: bool = False,
        class_ratio: Optional[float] = 1.0,
        max_samples: Optional[int] = None,
        include_originals: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.equal_methods = equal_methods
        self.class_ratio = class_ratio
        self.max_samples = max_samples
        self.include_originals = include_originals
        self.rng = random.Random(seed)

        # Index building
        self._original_indices: List[int] = []
        self._method_true: Dict[str, List[int]] = defaultdict(list)
        self._method_false: Dict[str, List[int]] = defaultdict(list)

        self._index_dataset()

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_true_label(label: str) -> bool:
        if label is None:
            return False
        low = label.lower().strip()
        return "true" in low and "false" not in low

    def _extract_label(self, example) -> Optional[str]:
        if "label" in example:
            return example["label"]
        last_msg = example["conversations"][-1]
        if last_msg["role"] == "assistant":
            return last_msg["content"]
        return None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_dataset(self):
        """Partition every row into original vs method×class buckets."""
        for idx in range(len(self.dataset)):
            ex = self.dataset[idx]

            if (
                ex["perturbation_type"]
                == PerturbationType.ORIGINAL_NORMALIZED.value
            ):
                self._original_indices.append(idx)
                continue

            method = ex.get("perturbation_method", "")
            label = self._extract_label(ex)
            if self._is_true_label(label):
                self._method_true[method].append(idx)
            else:
                self._method_false[method].append(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def balance(self) -> Dataset:
        """Return a balanced subset of the dataset."""
        self._print_header("INPUT DISTRIBUTION")
        self._print_input_stats()

        selected: List[int] = []

        # 1. Originals
        if self.include_originals:
            selected.extend(self._original_indices)

        # 2. Determine all methods present
        all_methods = sorted(
            set(self._method_true.keys()) | set(self._method_false.keys())
        )
        if not all_methods:
            print("  WARNING: No perturbed examples found.")
            return self._finalise(selected)

        # 3. Compute per-method budgets
        total_perturbed_available = sum(
            len(self._method_true[m]) + len(self._method_false[m])
            for m in all_methods
        )

        # Start with all perturbed samples as the budget ceiling
        perturbed_budget = total_perturbed_available
        if self.max_samples is not None:
            originals_used = len(selected)
            perturbed_budget = min(
                perturbed_budget, self.max_samples - originals_used
            )
            perturbed_budget = max(0, perturbed_budget)

        method_budgets = self._allocate_method_budgets(
            all_methods, perturbed_budget, total_perturbed_available
        )

        # 4. Within each method, split by class ratio
        for method in all_methods:
            budget = method_budgets[method]
            if budget <= 0:
                continue
            method_selected = self._select_for_method(method, budget)
            selected.extend(method_selected)

        return self._finalise(selected)

    # ------------------------------------------------------------------
    # Method budget allocation
    # ------------------------------------------------------------------

    def _allocate_method_budgets(
        self,
        methods: List[str],
        total_budget: int,
        total_available: int,
    ) -> Dict[str, int]:
        """Decide how many samples each method gets."""
        n = len(methods)
        if n == 0:
            return {}

        if self.equal_methods:
            return self._allocate_equal(methods, total_budget)
        else:
            return self._allocate_proportional(
                methods, total_budget, total_available
            )

    def _allocate_equal(
        self, methods: List[str], budget: int
    ) -> Dict[str, int]:
        """Give every method the same number of samples.

        The per-method cap is the *effective* capacity of the smallest
        method.  Effective capacity accounts for ``class_ratio``: e.g. if
        a method has 294 True and 218 False and ``class_ratio=1.0``, its
        effective capacity is ``218 + 218 = 436`` (capped by the minority
        class).

        No surplus redistribution — equal means equal.
        """
        n = len(methods)

        # Compute each method's effective capacity
        effective: Dict[str, int] = {}
        for m in methods:
            n_true = len(self._method_true[m])
            n_false = len(self._method_false[m])
            total = n_true + n_false

            if self.class_ratio is not None and n_true > 0 and n_false > 0:
                r = self.class_ratio
                # Capacity limited by minority class
                max_from_true = int(n_true * (1 + r) / r)
                max_from_false = int(n_false * (1 + r))
                effective[m] = min(total, max_from_true, max_from_false)
            else:
                effective[m] = total

        # Equal = smallest effective capacity
        per_method = min(effective.values())
        per_method = min(per_method, budget // n)

        avail = {
            m: len(self._method_true[m]) + len(self._method_false[m])
            for m in methods
        }
        alloc = {m: per_method for m in methods}

        self._log_allocations(methods, alloc, avail, mode="equal")
        return alloc

    def _allocate_proportional(
        self,
        methods: List[str],
        budget: int,
        total_available: int,
    ) -> Dict[str, int]:
        """Proportional allocation with a minimum-share guarantee.

        Every method gets at least ``floor(budget / n_methods)`` samples
        (capped by its actual size), then any leftover is distributed
        proportionally to methods that still have capacity.
        """
        n = len(methods)
        avail = {
            m: len(self._method_true[m]) + len(self._method_false[m])
            for m in methods
        }

        if total_available == 0:
            return {m: 0 for m in methods}

        # Minimum share guarantee
        min_share = budget // n

        alloc: Dict[str, int] = {}
        for m in methods:
            # Proportional ideal
            ideal = math.floor(budget * avail[m] / total_available)
            # At least min_share, at most what's available
            alloc[m] = min(max(ideal, min_share), avail[m])

        # If we over-allocated, trim proportionally
        total_alloc = sum(alloc.values())
        if total_alloc > budget:
            scale = budget / total_alloc
            alloc = {m: int(alloc[m] * scale) for m in methods}
            total_alloc = sum(alloc.values())

        # Distribute any remaining budget
        remaining = budget - total_alloc
        if remaining > 0:
            # Largest-remainder distribution
            fractionals = {}
            for m in methods:
                exact = budget * avail[m] / total_available
                fractionals[m] = exact - alloc[m]
            for m in sorted(fractionals, key=fractionals.get, reverse=True):
                if remaining <= 0:
                    break
                if alloc[m] < avail[m]:
                    alloc[m] += 1
                    remaining -= 1

        self._log_allocations(methods, alloc, avail, mode="proportional")
        return alloc

    # ------------------------------------------------------------------
    # Per-method class-balanced sampling
    # ------------------------------------------------------------------

    def _select_for_method(self, method: str, budget: int) -> List[int]:
        """Sample *budget* indices for one method, respecting class_ratio."""
        true_pool = self._method_true[method]
        false_pool = self._method_false[method]

        if self.class_ratio is None:
            # No class balancing — just sample uniformly
            combined = true_pool + false_pool
            return self._safe_sample(combined, budget)

        # Desired split
        # class_ratio = True / False  →  n_true = budget * r/(1+r)
        r = self.class_ratio
        n_true_ideal = budget * r / (1 + r)
        n_false_ideal = budget - n_true_ideal

        # Clamp to available
        n_true = min(round(n_true_ideal), len(true_pool))
        n_false = min(round(n_false_ideal), len(false_pool))

        # If one side is short, give the surplus to the other
        shortfall_true = round(n_true_ideal) - n_true
        shortfall_false = round(n_false_ideal) - n_false

        if shortfall_true > 0:
            n_false = min(n_false + shortfall_true, len(false_pool))
        if shortfall_false > 0:
            n_true = min(n_true + shortfall_false, len(true_pool))

        # Don't exceed budget
        while n_true + n_false > budget:
            if n_true > n_false:
                n_true -= 1
            else:
                n_false -= 1

        selected = self._safe_sample(true_pool, n_true)
        selected += self._safe_sample(false_pool, n_false)
        return selected

    def _safe_sample(self, pool: List[int], n: int) -> List[int]:
        if n <= 0:
            return []
        if n >= len(pool):
            return pool.copy()
        return self.rng.sample(pool, n)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def _finalise(self, selected: List[int]) -> Dataset:
        """Create the output dataset and print a summary."""
        if not selected:
            print("  WARNING: No examples selected. Returning empty dataset.")
            return Dataset.from_list([])

        balanced = self.dataset.select(selected)
        self._print_header("OUTPUT DISTRIBUTION")
        self._print_output_stats(balanced)
        return balanced

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(title: str):
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)

    def _print_input_stats(self):
        print(f"  Total examples : {len(self.dataset)}")
        print(f"  Originals      : {len(self._original_indices)}")
        all_methods = sorted(
            set(self._method_true.keys()) | set(self._method_false.keys())
        )
        print(f"  Methods found  : {all_methods}")
        print()
        header = f"  {'method':<16s}  {'True':>6s}  {'False':>6s}  {'Total':>6s}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for m in all_methods:
            nt = len(self._method_true[m])
            nf = len(self._method_false[m])
            print(f"  {m:<16s}  {nt:>6d}  {nf:>6d}  {nt+nf:>6d}")
        total_t = sum(len(v) for v in self._method_true.values())
        total_f = sum(len(v) for v in self._method_false.values())
        print(f"  {'TOTAL':<16s}  {total_t:>6d}  {total_f:>6d}  {total_t+total_f:>6d}")

    def _print_output_stats(self, dataset: Dataset):
        n = len(dataset)
        print(f"  Total examples: {n}")

        # Label distribution
        true_count = 0
        false_count = 0
        for ex in dataset:
            label = self._extract_label(ex)
            if self._is_true_label(label):
                true_count += 1
            else:
                false_count += 1
        if n > 0:
            print(
                f"  True : {true_count:>5d}  ({100*true_count/n:.1f}%)"
            )
            print(
                f"  False: {false_count:>5d}  ({100*false_count/n:.1f}%)"
            )

        # Per-method breakdown
        if "perturbation_method" in dataset.column_names:
            method_true: Dict[str, int] = defaultdict(int)
            method_false: Dict[str, int] = defaultdict(int)
            for ex in dataset:
                m = ex.get("perturbation_method", "")
                label = self._extract_label(ex)
                if self._is_true_label(label):
                    method_true[m] += 1
                else:
                    method_false[m] += 1

            all_methods = sorted(
                set(method_true.keys()) | set(method_false.keys())
            )
            print()
            header = (
                f"  {'method':<16s}  {'True':>6s}  {'False':>6s}"
                f"  {'Total':>6s}  {'%':>6s}"
            )
            print(header)
            print(f"  {'-' * (len(header) - 2)}")
            for m in all_methods:
                nt = method_true.get(m, 0)
                nf = method_false.get(m, 0)
                tot = nt + nf
                pct = 100 * tot / n if n else 0
                print(
                    f"  {m:<16s}  {nt:>6d}  {nf:>6d}"
                    f"  {tot:>6d}  {pct:>5.1f}%"
                )

    def _log_allocations(
        self,
        methods: List[str],
        alloc: Dict[str, int],
        avail: Dict[str, int],
        mode: str,
    ):
        print(f"\n  Method budget allocation ({mode}):")
        total_alloc = 0
        for m in methods:
            a = alloc.get(m, 0)
            v = avail.get(m, 0)
            total_alloc += a
            print(f"    {m:<16s}: {a:>5d} / {v:>5d}")
        print(f"    {'TOTAL':<16s}: {total_alloc:>5d}")
