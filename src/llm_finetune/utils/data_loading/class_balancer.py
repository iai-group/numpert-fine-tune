"""class_balancer.py."""

from datasets import Dataset
from typing import Dict, List
import random

from src.llm_finetune.utils.data_loading.perturbation_types import (
    PerturbationType,
)

# JSON format labels
LABEL_TRUE = '{"label": true}'
LABEL_FALSE = '{"label": false}'


class DatasetBalancer:
    """Balance dataset by label distribution and perturbation ratios.

    Key features:
    - Balance True/False labels
    - Control perturbed:non-perturbed ratio
    - Control T→F vs F→F ratio among false perturbations
    - Only uses structure-preserving (type1) perturbations for F→F
    """

    def __init__(
        self,
        dataset,
        balance_labels: bool = False,
        perturb_ratio: float = 1.0,
        false_perturb_t_to_f_ratio: float = 0.67,
        equal_perturbation_methods: bool = False,
        seed: int = 42,
    ):
        """Initialize the DatasetBalancer.

        Args:
            dataset: Full expanded dataset with all perturbations.
            balance_labels: Whether to balance True/False distribution.
            perturb_ratio: Ratio of perturbed to non-perturbed examples
                (1.0 = 1:1).
            false_perturb_t_to_f_ratio: Among False perturbed examples,
                ratio that should be T→F.
            equal_perturbation_methods: When True, stratify sampling so
                that each perturbation method (numeration, range, etc.)
                gets an equal share of the budget within each category.
                Requires the ``perturbation`` column in the dataset.
            seed: Random seed.
        """
        self.dataset = dataset
        self.balance_labels = balance_labels
        self.perturb_ratio = perturb_ratio
        self.false_perturb_t_to_f_ratio = false_perturb_t_to_f_ratio
        self.equal_perturbation_methods = equal_perturbation_methods
        self.rng = random.Random(seed)

        # Categorize examples
        self.original_true_indices: List[int] = []
        self.original_false_indices: List[int] = []
        self.t_to_t_indices: List[int] = []  # True → True perturbations
        self.t_to_f_indices: List[int] = []  # True → False perturbations
        self.f_to_f_indices: List[
            int
        ] = []  # False → False perturbations (*_true only)
        self.f_to_t_indices: List[int] = []  # False → True perturbations
        self.uncategorized_indices: List[
            int
        ] = []  # F→F non-type1 (type3 obfuscation on False originals)

        self._categorize_examples()

    def _is_true_label(self, label: str) -> bool:
        """Check if a label represents True (handles JSON format)."""
        if label is None:
            return False
        label_lower = label.lower().strip()
        return "true" in label_lower and "false" not in label_lower

    def _is_false_label(self, label: str) -> bool:
        """Check if a label represents False (handles JSON format)."""
        if label is None:
            return False
        label_lower = label.lower().strip()
        return "false" in label_lower

    def _categorize_examples(self):  # noqa: C901
        """Categorize all examples by type."""
        # First pass: identify original examples and their labels
        original_labels = (
            {}
        )  # claim_id -> original label (True/False as boolean)

        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            claim_id = example["claim_id"]
            perturb_type = example["perturbation_type"]

            if perturb_type == PerturbationType.ORIGINAL_NORMALIZED.value:
                label = self._extract_label(example)
                is_true = self._is_true_label(label)
                original_labels[claim_id] = is_true

                if is_true:
                    self.original_true_indices.append(idx)
                else:
                    self.original_false_indices.append(idx)

        # Second pass: categorize perturbations
        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            claim_id = example["claim_id"]
            perturb_type = example["perturbation_type"]

            if perturb_type == PerturbationType.ORIGINAL_NORMALIZED.value:
                continue

            original_is_true = original_labels.get(claim_id)
            current_label = self._extract_label(example)
            current_is_true = self._is_true_label(current_label)

            # Categorize based on original → current label transition
            if original_is_true and current_is_true:
                self.t_to_t_indices.append(idx)
            elif original_is_true and not current_is_true:
                self.t_to_f_indices.append(idx)
            elif not original_is_true and current_is_true:
                self.f_to_t_indices.append(idx)
            elif not original_is_true and not current_is_true:
                # Only use structure-preserving (type1) perturbations for F→F
                # (these are most similar to the original claim)
                if perturb_type == "perturb_type1":
                    self.f_to_f_indices.append(idx)
                else:
                    self.uncategorized_indices.append(idx)

    def _extract_label(self, example):
        """Extract label from example."""
        if "label" in example:
            return example["label"]
        else:
            # Extract from last assistant message
            last_msg = example["conversations"][-1]
            if last_msg["role"] == "assistant":
                return last_msg["content"]
        return None

    def balance(self):
        """Create balanced dataset according to specified ratios.

        Returns:
            Balanced dataset
        """
        print("\n" + "=" * 80)
        print("DATASET BALANCING")
        print("=" * 80)

        print("Configuration:")
        print(f"  Balance labels: {self.balance_labels}")
        print(f"  Perturb ratio: {self.perturb_ratio}")
        print(f"  False perturb T→F ratio: {self.false_perturb_t_to_f_ratio}")
        print(
            "  Equal perturbation methods:"
            f" {self.equal_perturbation_methods}"
        )

        print("\nOriginal distribution:")
        print(f"  Original True: {len(self.original_true_indices)}")
        print(f"  Original False: {len(self.original_false_indices)}")
        print(f"  T→T perturbations: {len(self.t_to_t_indices)}")
        print(f"  T→F perturbations: {len(self.t_to_f_indices)}")
        print(f"  F→T perturbations: {len(self.f_to_t_indices)}")
        print(f"  F→F perturbations (*_true only): {len(self.f_to_f_indices)}")
        print(
            f"  F→F uncategorized (non-type1): "
            f"{len(self.uncategorized_indices)}"
        )
        categorized = (
            len(self.original_true_indices)
            + len(self.original_false_indices)
            + len(self.t_to_t_indices)
            + len(self.t_to_f_indices)
            + len(self.f_to_t_indices)
            + len(self.f_to_f_indices)
            + len(self.uncategorized_indices)
        )
        excluded = len(self.dataset) - categorized
        print(f"  Total categorized: {categorized}")
        if excluded > 0:
            print(f"  ⚠️  Excluded from all categories: {excluded}")

        # Handle empty dataset
        if len(self.dataset) == 0:
            print("\n⚠️  WARNING: Empty dataset! Returning empty dataset.")
            print("=" * 80 + "\n")
            return self.dataset

        # Step 1: Apply perturb_ratio to get target counts
        n_original_true = len(self.original_true_indices)
        n_original_false = len(self.original_false_indices)

        # Calculate perturbation targets based on perturb_ratio
        n_target_t_to_t = int(n_original_true * self.perturb_ratio)
        n_target_false_perturbed = int(n_original_false * self.perturb_ratio)

        # Split false perturbations between T→F and F→F
        n_target_t_to_f = int(
            n_target_false_perturbed * self.false_perturb_t_to_f_ratio
        )
        n_target_f_to_f = n_target_false_perturbed - n_target_t_to_f

        # Initial targets before balancing
        initial_true_total = n_original_true + n_target_t_to_t
        initial_false_total = (
            n_original_false + n_target_t_to_f + n_target_f_to_f
        )

        print(f"\nAfter perturb_ratio ({self.perturb_ratio}):")
        print(
            f"  True: {n_original_true} original"
            f" + {n_target_t_to_t} T→T"
            f" = {initial_true_total} total"
        )
        print(
            f"  False: {n_original_false} original"
            f" + {n_target_t_to_f} T→F"
            f" + {n_target_f_to_f} F→F"
            f" = {initial_false_total} total"
        )

        # Step 2: Apply label balancing if requested
        # (downsample majority class proportionally)
        if self.balance_labels:
            # Handle edge case where one class has zero examples
            if initial_true_total == 0 or initial_false_total == 0:
                print(
                    "\n⚠️  WARNING: Cannot balance labels"
                    " when one class has 0 examples!"
                )
                print(
                    f"     True total: {initial_true_total},"
                    f" False total: {initial_false_total}"
                )
                print("     Proceeding without balancing.")
            else:
                min_total = min(initial_true_total, initial_false_total)

                if initial_true_total > initial_false_total:
                    # Downsample True proportionally
                    print(
                        f"\n⚙️  Downsampling True class"
                        f" from {initial_true_total} to {min_total}"
                    )
                    downsample_ratio = min_total / initial_true_total
                    n_original_true = int(n_original_true * downsample_ratio)
                    n_target_t_to_t = int(n_target_t_to_t * downsample_ratio)

                elif initial_false_total > initial_true_total:
                    # Downsample False proportionally
                    print(
                        f"\n⚙️  Downsampling False class"
                        f" from {initial_false_total} to {min_total}"
                    )
                    downsample_ratio = min_total / initial_false_total
                    n_original_false = int(n_original_false * downsample_ratio)
                    n_target_t_to_f = int(n_target_t_to_f * downsample_ratio)
                    n_target_f_to_f = int(n_target_f_to_f * downsample_ratio)

        print("\nFinal target distribution:")
        print(f"  Original True: {n_original_true}")
        print(f"  Original False: {n_original_false}")
        print(f"  T→T perturbations: {n_target_t_to_t}")
        print(f"  T→F perturbations: {n_target_t_to_f}")
        print(f"  F→F perturbations: {n_target_f_to_f}")

        final_true = n_original_true + n_target_t_to_t
        final_false = n_original_false + n_target_t_to_f + n_target_f_to_f
        final_total = final_true + final_false

        # Safe percentage calculation
        if final_total > 0:
            print(
                f"  True total: {final_true}"
                f" ({100*final_true/final_total:.1f}%)"
            )
            print(
                f"  False total: {final_false}"
                f" ({100*final_false/final_total:.1f}%)"
            )
        else:
            print(f"  True total: {final_true}")
            print(f"  False total: {final_false}")

        print(f"  Grand total: {final_total}")

        # Sample indices
        selected_indices = []

        # Sample originals
        selected_indices.extend(
            self._sample_with_warning(
                self.original_true_indices, n_original_true, "Original True"
            )
        )
        selected_indices.extend(
            self._sample_with_warning(
                self.original_false_indices, n_original_false, "Original False"
            )
        )

        # Sample perturbations
        selected_indices.extend(
            self._sample_with_warning(
                self.t_to_t_indices, n_target_t_to_t, "T→T"
            )
        )
        selected_indices.extend(
            self._sample_with_warning(
                self.t_to_f_indices, n_target_t_to_f, "T→F"
            )
        )
        selected_indices.extend(
            self._sample_with_warning(
                self.f_to_f_indices, n_target_f_to_f, "F→F (*_true only)"
            )
        )

        # Handle case where no indices were selected
        if len(selected_indices) == 0:
            print(
                "\n⚠️  WARNING: No indices selected!"
                " Returning empty dataset."
            )
            print("=" * 80 + "\n")
            return Dataset.from_list([])

        # Create balanced dataset
        balanced_dataset = self.dataset.select(selected_indices)

        # Verify final distribution
        actual_true = sum(
            1
            for idx in selected_indices
            if self._is_true_label(self._extract_label(self.dataset[idx]))
        )
        actual_false = len(selected_indices) - actual_true

        print("\nActual final label distribution:")
        if len(selected_indices) > 0:
            print(
                f"  True: {actual_true}"
                f" ({100*actual_true/len(selected_indices):.1f}%)"
            )
            print(
                f"  False: {actual_false}"
                f" ({100*actual_false/len(selected_indices):.1f}%)"
            )
        else:
            print(f"  True: {actual_true}")
            print(f"  False: {actual_false}")
        print("=" * 80 + "\n")

        return balanced_dataset

    def _sample_with_warning(
        self, indices: List[int], target: int, category: str
    ) -> List[int]:
        """Sample indices with optional stratification.

        Stratification modes (checked in order):

        1. **Equal-method** (``equal_perturbation_methods=True``):
           Groups indices by the ``perturbation`` column (e.g. numeration,
           range, …) and allocates an equal share of the budget to each
           method.  Any method with fewer examples than its share donates
           the surplus back to the others via a second pass.

        2. **Proportional-source** (default when ``source_file`` exists):
           Groups by ``source_file`` and allocates proportionally.

        3. **Uniform random** (fallback): plain ``random.sample``.
        """
        available = len(indices)

        if target <= 0:
            return []

        if available <= target:
            if available < target:
                print(f"  ⚠️  WARNING: Not enough {category} examples!")
                print(f"     Requested: {target}, Available: {available}")
                print(f"     Using all {available} available examples.")
            return indices.copy()

        # ----- equal-method stratification --------------------------------
        if self.equal_perturbation_methods:
            has_method = "perturbation" in self.dataset.column_names
            if has_method:
                return self._stratify_equal(
                    indices, target, category, group_col="perturbation"
                )

        # ----- proportional source_file stratification --------------------
        has_source = "source_file" in self.dataset.column_names
        if not has_source:
            return self.rng.sample(indices, target)

        return self._stratify_proportional(
            indices, target, category, group_col="source_file"
        )

    # ------------------------------------------------------------------
    # Stratification helpers
    # ------------------------------------------------------------------

    def _build_groups(
        self, indices: List[int], group_col: str
    ) -> Dict[str, List[int]]:
        """Group *indices* by a dataset column value."""
        groups: Dict[str, List[int]] = {}
        for idx in indices:
            key = self.dataset[idx][group_col]
            groups.setdefault(key, []).append(idx)
        return groups

    def _stratify_equal(
        self,
        indices: List[int],
        target: int,
        category: str,
        group_col: str,
    ) -> List[int]:
        """Allocate *target* slots equally across groups.

        Each group gets ``target // n_groups`` slots (capped by its size).
        Left-over budget is distributed round-robin to groups that still
        have capacity.
        """
        groups = self._build_groups(indices, group_col)
        n_groups = len(groups)

        if n_groups <= 1:
            return self.rng.sample(indices, target)

        # Equal base allocation
        base_per_group = target // n_groups
        allocations: Dict[str, int] = {
            g: min(base_per_group, len(idxs)) for g, idxs in groups.items()
        }

        # Distribute remainder — groups with room get extra
        remaining = target - sum(allocations.values())
        if remaining > 0:
            for g in sorted(groups):
                if remaining <= 0:
                    break
                room = len(groups[g]) - allocations[g]
                extra = min(room, remaining)
                allocations[g] += extra
                remaining -= extra

        selected = self._select_from_groups(groups, allocations)
        self._log_stratification(category, groups, allocations, label="method")
        return selected

    def _stratify_proportional(
        self,
        indices: List[int],
        target: int,
        category: str,
        group_col: str,
    ) -> List[int]:
        """Allocate *target* slots proportionally across groups.

        Uses the largest-remainder method for rounding.
        """
        groups = self._build_groups(indices, group_col)
        available = len(indices)

        if len(groups) <= 1:
            return self.rng.sample(indices, target)

        allocations: Dict[str, int] = {}
        for g, idxs in groups.items():
            exact = target * len(idxs) / available
            allocations[g] = min(int(exact), len(idxs))

        remaining = target - sum(allocations.values())
        if remaining > 0:
            fractionals = {
                g: (target * len(groups[g]) / available) - allocations[g]
                for g in groups
            }
            for g in sorted(fractionals, key=fractionals.get, reverse=True):
                if remaining <= 0:
                    break
                if allocations[g] < len(groups[g]):
                    allocations[g] += 1
                    remaining -= 1

        selected = self._select_from_groups(groups, allocations)
        self._log_stratification(category, groups, allocations, label="source")
        return selected

    def _select_from_groups(
        self,
        groups: Dict[str, List[int]],
        allocations: Dict[str, int],
    ) -> List[int]:
        """Sample *allocations[g]* indices from each group."""
        selected: List[int] = []
        for g, idxs in groups.items():
            n = allocations.get(g, 0)
            if n <= 0:
                continue
            if n >= len(idxs):
                selected.extend(idxs)
            else:
                selected.extend(self.rng.sample(idxs, n))
        return selected

    def _log_stratification(
        self,
        category: str,
        groups: Dict[str, List[int]],
        allocations: Dict[str, int],
        label: str,
    ) -> None:
        """Print a breakdown of the stratified allocation."""
        print(f"  {category} stratified by {label}:")
        for g in sorted(groups):
            avail = len(groups[g])
            sampled = allocations.get(g, 0)
            print(f"    {g:45s}: {sampled:5d}/{avail:5d}")
