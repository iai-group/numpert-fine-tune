"""
Dataset balancing utilities - no CUDA dependencies
"""

from datasets import Dataset
from typing import List
from collections import defaultdict
import random


class DatasetBalancer:
    """
    Balance dataset by label distribution and perturbation ratios.
    
    Key features:
    - Balance True/False labels
    - Control perturbed:non-perturbed ratio
    - Control T→F vs F→F ratio among false perturbations
    - Only uses *_true perturbations for F→F (more similar to original)
    """
    
    def __init__(
        self,
        dataset,
        balance_labels: bool = False,
        perturb_ratio: float = 1.0,
        false_perturb_t_to_f_ratio: float = 0.67,
        seed: int = 42
    ):
        """
        Args:
            dataset: Full expanded dataset with all perturbations
            balance_labels: Whether to balance True/False distribution
            perturb_ratio: Ratio of perturbed to non-perturbed examples (1.0 = 1:1)
            false_perturb_t_to_f_ratio: Among False perturbed examples, ratio that should be T→F
            seed: Random seed
        """
        self.dataset = dataset
        self.balance_labels = balance_labels
        self.perturb_ratio = perturb_ratio
        self.false_perturb_t_to_f_ratio = false_perturb_t_to_f_ratio
        self.rng = random.Random(seed)
        
        # Categorize examples
        self.original_true_indices = []
        self.original_false_indices = []
        self.t_to_t_indices = []  # True → True perturbations
        self.t_to_f_indices = []  # True → False perturbations
        self.f_to_f_indices = []  # False → False perturbations (*_true only)
        
        self._categorize_examples()
        
    def _categorize_examples(self):
        """Categorize all examples by type"""
        # First pass: identify original examples and their labels
        original_labels = {}  # claim_id -> original label
        
        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            claim_id = example['claim_id']
            perturb_type = example['perturbation_type']
            
            if perturb_type == 'original':
                label = self._extract_label(example)
                original_labels[claim_id] = label
                
                if label == "True":
                    self.original_true_indices.append(idx)
                else:
                    self.original_false_indices.append(idx)
        
        # Second pass: categorize perturbations
        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            claim_id = example['claim_id']
            perturb_type = example['perturbation_type']
            
            if perturb_type == 'original':
                continue
            
            original_label = original_labels.get(claim_id)
            current_label = self._extract_label(example)
            
            # Categorize based on original → current label transition
            if original_label == "True" and current_label == "True":
                self.t_to_t_indices.append(idx)
            elif original_label == "True" and current_label == "False":
                self.t_to_f_indices.append(idx)
            elif original_label == "False" and current_label == "False":
                # Only use *_true perturbations for F→F (more similar to original)
                if perturb_type.endswith('_true'):
                    self.f_to_f_indices.append(idx)
                # Ignore *_false perturbations for F→F
    
    def _extract_label(self, example):
        """Extract label from example"""
        if 'ground_truth' in example:
            return example['ground_truth']
        else:
            # Extract from last assistant message
            last_msg = example['conversations'][-1]
            if last_msg['role'] == 'assistant':
                return last_msg['content']
        return None
    
    def balance(self):
        """
        Create balanced dataset according to specified ratios.
        
        Returns:
            Balanced dataset
        """
        print("\n" + "="*80)
        print("DATASET BALANCING")
        print("="*80)
        
        print(f"Configuration:")
        print(f"  Balance labels: {self.balance_labels}")
        print(f"  Perturb ratio: {self.perturb_ratio}")
        print(f"  False perturb T→F ratio: {self.false_perturb_t_to_f_ratio}")
        
        print(f"\nOriginal distribution:")
        print(f"  Original True: {len(self.original_true_indices)}")
        print(f"  Original False: {len(self.original_false_indices)}")
        print(f"  T→T perturbations: {len(self.t_to_t_indices)}")
        print(f"  T→F perturbations: {len(self.t_to_f_indices)}")
        print(f"  F→F perturbations (*_true only): {len(self.f_to_f_indices)}")
        print(f"  Total available: {len(self.dataset)}")
        
        # Handle empty dataset
        if len(self.dataset) == 0:
            print("\n⚠️  WARNING: Empty dataset! Returning empty dataset.")
            print("="*80 + "\n")
            return self.dataset
        
        # Step 1: Apply perturb_ratio to get target counts
        n_original_true = len(self.original_true_indices)
        n_original_false = len(self.original_false_indices)
        
        # Calculate perturbation targets based on perturb_ratio
        n_target_t_to_t = int(n_original_true * self.perturb_ratio)
        n_target_false_perturbed = int(n_original_false * self.perturb_ratio)
        
        # Split false perturbations between T→F and F→F
        n_target_t_to_f = int(n_target_false_perturbed * self.false_perturb_t_to_f_ratio)
        n_target_f_to_f = n_target_false_perturbed - n_target_t_to_f
        
        # Initial targets before balancing
        initial_true_total = n_original_true + n_target_t_to_t
        initial_false_total = n_original_false + n_target_t_to_f + n_target_f_to_f
        
        print(f"\nAfter perturb_ratio ({self.perturb_ratio}):")
        print(f"  True: {n_original_true} original + {n_target_t_to_t} T→T = {initial_true_total} total")
        print(f"  False: {n_original_false} original + {n_target_t_to_f} T→F + {n_target_f_to_f} F→F = {initial_false_total} total")
        
        # Step 2: Apply label balancing if requested (downsample majority class proportionally)
        if self.balance_labels:
            # Handle edge case where one class has zero examples
            if initial_true_total == 0 or initial_false_total == 0:
                print(f"\n⚠️  WARNING: Cannot balance labels when one class has 0 examples!")
                print(f"     True total: {initial_true_total}, False total: {initial_false_total}")
                print(f"     Proceeding without balancing.")
            else:
                min_total = min(initial_true_total, initial_false_total)
                
                if initial_true_total > initial_false_total:
                    # Downsample True proportionally
                    print(f"\n⚙️  Downsampling True class from {initial_true_total} to {min_total}")
                    downsample_ratio = min_total / initial_true_total
                    n_original_true = int(n_original_true * downsample_ratio)
                    n_target_t_to_t = int(n_target_t_to_t * downsample_ratio)
                    
                elif initial_false_total > initial_true_total:
                    # Downsample False proportionally
                    print(f"\n⚙️  Downsampling False class from {initial_false_total} to {min_total}")
                    downsample_ratio = min_total / initial_false_total
                    n_original_false = int(n_original_false * downsample_ratio)
                    n_target_t_to_f = int(n_target_t_to_f * downsample_ratio)
                    n_target_f_to_f = int(n_target_f_to_f * downsample_ratio)
        
        print(f"\nFinal target distribution:")
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
            print(f"  True total: {final_true} ({100*final_true/final_total:.1f}%)")
            print(f"  False total: {final_false} ({100*final_false/final_total:.1f}%)")
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
            print(f"\n⚠️  WARNING: No indices selected! Returning empty dataset.")
            print("="*80 + "\n")
            return Dataset.from_list([])
        
        # Create balanced dataset
        balanced_dataset = self.dataset.select(selected_indices)
        
        # Verify final distribution
        actual_true = sum(1 for idx in selected_indices 
                         if self._extract_label(self.dataset[idx]) == "True")
        actual_false = len(selected_indices) - actual_true
        
        print(f"\nActual final label distribution:")
        if len(selected_indices) > 0:
            print(f"  True: {actual_true} ({100*actual_true/len(selected_indices):.1f}%)")
            print(f"  False: {actual_false} ({100*actual_false/len(selected_indices):.1f}%)")
        else:
            print(f"  True: {actual_true}")
            print(f"  False: {actual_false}")
        print("="*80 + "\n")
        
        return balanced_dataset
    
    def _sample_with_warning(self, indices: List[int], target: int, category: str) -> List[int]:
        """Sample indices with warning if not enough available"""
        available = len(indices)
        
        if target <= 0:
            return []
        
        if available < target:
            print(f"  ⚠️  WARNING: Not enough {category} examples!")
            print(f"     Requested: {target}, Available: {available}")
            print(f"     Using all {available} available examples.")
            return indices.copy()
        
        return self.rng.sample(indices, target)