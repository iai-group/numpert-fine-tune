"""Batch sampling utilities for perturbation-aware batching."""

from torch.utils.data import Sampler
from collections import defaultdict
from typing import Any, Dict, List, Optional
import random


class EnhancedPerturbationBatchSampler(Sampler):
    """Enhanced batch sampler with claim_id matching and original+perturbed
    guarantees.

    Features:
    - Can group examples with same claim_id in batches
    - Ensures each batch has both original and perturbed examples
    - Splits large claims across batches with balanced perturbation mix
    - Never repeats the original claim across batches
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batch_match_claim_id: bool = False,
        split_large_claims: bool = True,
        ensure_original_in_batch: bool = True,
        ensure_perturbed_in_batch: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: HuggingFace Dataset with 'claim_id' and 'perturbation_type' columns
            batch_size: Number of examples per batch
            batch_match_claim_id: Try to group same claim_id in batches
            split_large_claims: Split claims with >batch_size examples across batches
            ensure_original_in_batch: Ensure each batch has at least one original
            ensure_perturbed_in_batch: Ensure each batch has at least one perturbed
            shuffle: Whether to shuffle the data
            seed: Random seed
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_match_claim_id = batch_match_claim_id
        self.split_large_claims = split_large_claims
        self.ensure_original_in_batch = ensure_original_in_batch
        self.ensure_perturbed_in_batch = ensure_perturbed_in_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        # Group indices
        self.claim_id_groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"original": None, "perturbed": []}
        )
        self.original_indices: List[int] = []
        self.perturbed_indices: List[int] = []

        for idx in range(len(dataset)):
            claim_id = dataset[idx]["claim_id"]
            perturb_type = dataset[idx]["perturbation_type"]

            if perturb_type == "original":
                self.claim_id_groups[claim_id]["original"] = idx
                self.original_indices.append(idx)
            else:
                self.claim_id_groups[claim_id]["perturbed"].append(idx)
                self.perturbed_indices.append(idx)

        print(f"\n{'='*80}")
        print(f"EnhancedPerturbationBatchSampler initialized:")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Unique claim_ids: {len(self.claim_id_groups)}")
        print(f"  Original examples: {len(self.original_indices)}")
        print(f"  Perturbed examples: {len(self.perturbed_indices)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batch match claim_id: {batch_match_claim_id}")
        print(f"  Split large claims: {split_large_claims}")
        print(f"  Ensure original in batch: {ensure_original_in_batch}")
        print(f"  Ensure perturbed in batch: {ensure_perturbed_in_batch}")
        print(f"  Drop last: {drop_last}")
        print(f"{'='*80}\n")

    def __iter__(self):
        if self.batch_match_claim_id:
            batches = self._create_claim_matched_batches()
        else:
            batches = self._create_mixed_batches()

        # Apply constraints if requested
        if self.ensure_original_in_batch or self.ensure_perturbed_in_batch:
            batches = self._enforce_batch_constraints(batches)

        if self.shuffle:
            self.rng.shuffle(batches)

        for batch in batches:
            yield batch

    def _create_claim_matched_batches(self):
        """Create batches where same claim_id examples are grouped together."""
        batches = []
        claim_ids = list(self.claim_id_groups.keys())

        if self.shuffle:
            self.rng.shuffle(claim_ids)

        for claim_id in claim_ids:
            group = self.claim_id_groups[claim_id]
            original_idx = group["original"]
            perturbed_indices = group["perturbed"].copy()

            if self.shuffle:
                self.rng.shuffle(perturbed_indices)

            # All indices for this claim (original comes first, then perturbations)
            claim_indices = []
            if original_idx is not None:
                claim_indices.append(original_idx)
            claim_indices.extend(perturbed_indices)

            # If claim fits in one batch, add it
            if len(claim_indices) <= self.batch_size:
                batches.append(claim_indices)

            # If claim is too large and splitting is enabled
            elif self.split_large_claims:
                # Split into multiple batches with balanced mix
                # IMPORTANT: Original only goes in the first batch, never repeated

                remaining_perturbed = perturbed_indices.copy()
                original_used = False

                while remaining_perturbed:
                    batch = []

                    # Add original to first batch only (never repeat)
                    if not original_used and original_idx is not None:
                        batch.append(original_idx)
                        original_used = True

                    # Fill rest of batch with perturbations
                    slots_available = self.batch_size - len(batch)
                    batch.extend(remaining_perturbed[:slots_available])
                    remaining_perturbed = remaining_perturbed[slots_available:]

                    batches.append(batch)

            # If splitting disabled, just take first batch_size examples
            else:
                batches.append(claim_indices[: self.batch_size])

        # Drop last incomplete batch if requested
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]

        return batches

    def _create_mixed_batches(self):
        """Create batches with any original + any perturbed."""
        all_indices = list(range(len(self.dataset)))

        if self.shuffle:
            self.rng.shuffle(all_indices)

        batches = []

        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i : i + self.batch_size]

            # Skip if dropping last and batch is incomplete
            if self.drop_last and len(batch) < self.batch_size:
                continue

            batches.append(batch)

        return batches

    def _enforce_batch_constraints(
        self, batches: List[List[int]]
    ) -> List[List[int]]:
        """Enforce constraints that batches should have both original and
        perturbed examples.

        This is done by swapping indices between batches to meet the
        constraints.
        """
        if not batches:
            return batches

        fixed_batches = []
        batches_needing_original = []
        batches_needing_perturbed = []
        batches_with_extra_original = []
        batches_with_extra_perturbed = []

        # First pass: categorize batches
        for batch_idx, batch in enumerate(batches):
            has_original = any(
                self.dataset[idx]["perturbation_type"] == "original"
                for idx in batch
            )
            has_perturbed = any(
                self.dataset[idx]["perturbation_type"] != "original"
                for idx in batch
            )

            # Count originals and perturbed
            original_count = sum(
                1
                for idx in batch
                if self.dataset[idx]["perturbation_type"] == "original"
            )
            perturbed_count = len(batch) - original_count

            needs_fixing = False

            if self.ensure_original_in_batch and not has_original:
                batches_needing_original.append((batch_idx, batch))
                needs_fixing = True
            elif original_count > 1:
                # Batch has extra originals that could be swapped
                batches_with_extra_original.append(
                    (batch_idx, batch, original_count)
                )

            if self.ensure_perturbed_in_batch and not has_perturbed:
                batches_needing_perturbed.append((batch_idx, batch))
                needs_fixing = True
            elif perturbed_count > 1:
                # Batch has extra perturbed that could be swapped
                batches_with_extra_perturbed.append(
                    (batch_idx, batch, perturbed_count)
                )

            if not needs_fixing:
                fixed_batches.append(batch)

        # Second pass: try to fix batches by swapping
        # Fix batches needing originals
        for batch_idx, batch in batches_needing_original:
            fixed = False

            # Try to swap with a batch that has extra originals
            for (
                extra_idx,
                extra_batch,
                orig_count,
            ) in batches_with_extra_original:
                if orig_count > 1:  # Has spare original
                    # Find an original in the extra batch
                    original_to_swap = None
                    for idx in extra_batch:
                        if self.dataset[idx]["perturbation_type"] == "original":
                            original_to_swap = idx
                            break

                    # Find a perturbed in the needing batch
                    perturbed_to_swap = None
                    for idx in batch:
                        if self.dataset[idx]["perturbation_type"] != "original":
                            perturbed_to_swap = idx
                            break

                    if (
                        original_to_swap is not None
                        and perturbed_to_swap is not None
                    ):
                        # Perform swap
                        batch.remove(perturbed_to_swap)
                        batch.append(original_to_swap)
                        extra_batch.remove(original_to_swap)
                        extra_batch.append(perturbed_to_swap)
                        fixed = True
                        break

            if not fixed:
                # Couldn't fix, but still add the batch with a warning
                print(
                    f"  ⚠️  Warning: Batch {batch_idx} has no original example (couldn't fix)"
                )

            fixed_batches.append(batch)

        # Fix batches needing perturbed
        for batch_idx, batch in batches_needing_perturbed:
            fixed = False

            # Try to swap with a batch that has extra perturbed
            for (
                extra_idx,
                extra_batch,
                pert_count,
            ) in batches_with_extra_perturbed:
                if pert_count > 1:  # Has spare perturbed
                    # Find a perturbed in the extra batch
                    perturbed_to_swap = None
                    for idx in extra_batch:
                        if self.dataset[idx]["perturbation_type"] != "original":
                            perturbed_to_swap = idx
                            break

                    # Find an original in the needing batch
                    original_to_swap = None
                    for idx in batch:
                        if self.dataset[idx]["perturbation_type"] == "original":
                            original_to_swap = idx
                            break

                    if (
                        perturbed_to_swap is not None
                        and original_to_swap is not None
                    ):
                        # Perform swap
                        batch.remove(original_to_swap)
                        batch.append(perturbed_to_swap)
                        extra_batch.remove(perturbed_to_swap)
                        extra_batch.append(original_to_swap)
                        fixed = True
                        break

            if not fixed:
                # Couldn't fix, but still add the batch with a warning
                print(
                    f"  ⚠️  Warning: Batch {batch_idx} has no perturbed example (couldn't fix)"
                )

            fixed_batches.append(batch)

        return fixed_batches

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
