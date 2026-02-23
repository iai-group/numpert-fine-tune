"""Main script to apply perturbations on numerical claims in a dataset."""

import code.data_preprocessing.perturbations.approximation as approximation
import code.data_preprocessing.perturbations.negative as negative
import code.data_preprocessing.perturbations.numeration as numeration
import code.data_preprocessing.perturbations.obfuscate as obfuscate
import code.data_preprocessing.perturbations.range as range_pert
from typing import Any, List, Dict
from code.data_preprocessing.perturbations.utils.load_save import (
    load_data,
    save_data,
)

INPUT_TRUE = "data/raw_data/binary_data/spacy_tagged/test_true_claims_tagged_normalized_numbers.json"
INPUT_FALSE = "data/raw_data/binary_data/spacy_tagged/test_false_claims_tagged_normalized_numbers.json"
OUTPUT_DIR = "data/perturbed_data/test_split"


def remove_unperturbed_claims(
    data: List[Dict[str, Any]], perturbation_keys: List[str]
) -> List[Dict[str, Any]]:
    """Remove claims where perturbations are identical to the normalized claim.

    Args:
        data: List of dictionaries containing claims.
        perturbation_keys: List of keys to check for perturbations.

    Returns:
        Filtered list with unperturbed claims removed.
    """
    filtered_data = []
    removed_count = 0

    for item in data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item: {type(item)}")
            continue

        normalized_claim = item.get("normalized_claim", "")
        if any(item.get(key) == normalized_claim for key in perturbation_keys if key in item):
            removed_count += 1
        else:
            filtered_data.append(item)

    if removed_count > 0:
        print(f"Removed {removed_count} unperturbed claims")

    return filtered_data


def save_splits(
    results: Dict[str, List[Dict[str, Any]]],
    name: str,
    perturbation_keys: List[str],
) -> None:
    """Apply fail-safe and save true/false splits for a perturbation module.

    Args:
        results: Dict with 'true' and 'false' splits from process_json.
        name: Perturbation name used in the output filename (e.g. 'numeration').
        perturbation_keys: Keys to check in the fail-safe filter.
    """
    for label, split_data in results.items():
        split_data = remove_unperturbed_claims(split_data, perturbation_keys)
        if split_data:
            path = f"{OUTPUT_DIR}/{name}_{label}.json"
            save_data(split_data, path)
            print(f"  Saved {len(split_data)} entries to: {path}")


def split_by_label(
    data: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Split a flat list of entries into true/false dicts by label."""
    results: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        label = entry.get("label", "UNKNOWN")
        results.setdefault(label, []).append(entry)
    return results

    

def main() -> None:
    """Main function to execute all perturbation processes."""
    data_true: List[Dict[str, Any]] = load_data(INPUT_TRUE)
    data_false: List[Dict[str, Any]] = load_data(INPUT_FALSE)
    data_all = data_true + data_false

    # Approximation — file-based interface, pass true/false separately
    print("Running approximation perturbations...")
    approximation.process_claims(INPUT_TRUE, f"{OUTPUT_DIR}/approximation_True.json")
    approximation.process_claims(INPUT_FALSE, f"{OUTPUT_DIR}/approximation_False.json")

    

# Negative number perturbations
    print("Running negative number perturbations...")
    save_splits(
        split_by_label(negative.process_json(data_all)),
        name="negative",
        perturbation_keys=["perturb_type2"],
    )

    # Numeration perturbations
    print("Running numeration perturbations...")
    save_splits(
        split_by_label(numeration.process_json(data_all)),
        name="numeration",
        perturbation_keys=["perturb_type1", "perturb_type2"],
    )

    # Range perturbations
    print("Running range perturbations...")
    save_splits(
        split_by_label(range_pert.process_json(data_all)),
        name="range",
        perturbation_keys=["perturb_type1", "perturb_type2"],
    )
    # Obfuscation perturbations
    print("Running obfuscation perturbations...")
    save_splits(
        obfuscate.process_json(data_all),
        name="obfuscation",
        perturbation_keys=["perturb_type3_random", "perturb_type3_masked", "perturb_type3_removed"],
    )



if __name__ == "__main__":
    try:
        main()
        print("\nAll perturbation processes completed successfully.")
    except Exception as e:
        print(f"Error: {e}")