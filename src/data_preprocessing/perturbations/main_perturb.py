"""Main script to apply perturbations on numerical claims in a dataset."""

import code.data_preprocessing.perturbations.approximation as approximation
import code.data_preprocessing.perturbations.negative as negative
import code.data_preprocessing.perturbations.numeration as numeration
import code.data_preprocessing.perturbations.obfuscate as obfuscate
import code.data_preprocessing.perturbations.range as range
from typing import Any, List, Dict
from code.data_preprocessing.perturbations.utils.load_save import (
    load_data,
    save_data,
)


def remove_unperturbed_claims(
    data: List[Dict[str, Any]], perturbation_keys: List[str]
) -> List[Dict[str, Any]]:
    """Remove claims where perturbations are identical to the normalized claim.

    Args:
        data: List of dictionaries containing claims
        perturbation_keys: List of keys to check for perturbations

    Returns:
        Filtered list with unperturbed claims removed
    """
    # Handle case where data might be nested in a list
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        # If data is a list of lists, flatten it
        flattened_data = []
        for sublist in data:
            if isinstance(sublist, list):
                flattened_data.extend(sublist)
            else:
                flattened_data.append(sublist)
        data = flattened_data

    filtered_data = []
    removed_count = 0

    for item in data:
        # Skip if item is not a dictionary
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item: {type(item)}")
            continue

        normalized_claim = item.get("normalized_claim", "")
        should_keep = True

        # Check if any perturbation key has the same value as normalized_claim
        for key in perturbation_keys:
            if key in item and item[key] == normalized_claim:
                should_keep = False
                break

        if should_keep:
            filtered_data.append(item)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"Removed {removed_count} unperturbed claims from dataset")

    return filtered_data


def main() -> None:
    """Main function to execute all perturbation processes."""
    # Run each perturbation module
    data_true: List[Dict[str, Any]] = load_data(
        "data/raw_data/binary_data/spacy_tagged/train_true_claims_tagged_normalized_numbers.json"
    )
    data_false: List[Dict[str, Any]] = load_data(
        "data/raw_data/binary_data/spacy_tagged/train_false_claims_tagged_normalized_numbers.json"
    )

    # Approximation perturbations_train
    approximation.process_claims(
        "data/raw_data/binary_data/spacy_tagged/train_true_claims_tagged_normalized_numbers.json",
        "data/perturbed_data/train_split/approximation_true.json",
    )
    approximation.process_claims(
        "data/raw_data/binary_data/spacy_tagged/train_false_claims_tagged_normalized_numbers.json",
        "data/perturbed_data/train_split/approximation_false.json",
    )

    # Negative number perturbations_train
    neg_true: List[Dict[str, Any]] = negative.process_json(data_true)
    neg_false: List[Dict[str, Any]] = negative.process_json(data_false)

    # Apply fail-safe for negative
    neg_true = remove_unperturbed_claims(
        neg_true, ["negative_true", "negative_false"]
    )
    neg_false = remove_unperturbed_claims(
        neg_false, ["negative_true", "negative_false"]
    )

    save_data(neg_true, "data/perturbed_data/train_split/negative_true.json")
    save_data(neg_false, "data/perturbed_data/train_split/negative_false.json")
    print("Negative number perturbations completed.")

    # Numeration perturbations
    num_true: List[Dict[str, Any]] = numeration.process_json(data_true)
    num_false: List[Dict[str, Any]] = numeration.process_json(data_false)

    # Apply fail-safe for numeration
    num_true = remove_unperturbed_claims(
        num_true, ["numeration_true", "numeration_false"]
    )
    num_false = remove_unperturbed_claims(
        num_false, ["numeration_true", "numeration_false"]
    )

    save_data(num_true, "data/perturbed_data/train_split/numeration_true.json")
    save_data(
        num_false, "data/perturbed_data/train_split/numeration_false.json"
    )
    print("Numeration perturbations completed.")

    # The function returns a tuple of three lists
    (
        random_data_true,
        masking_data_true,
        removal_data_true,
    ) = obfuscate.process_json(data_true)
    (
        random_data_false,
        masking_data_false,
        removal_data_false,
    ) = obfuscate.process_json(data_false)

    # Combine the lists into single lists for true and false perturbations
    obf_true = random_data_true + masking_data_true + removal_data_true
    obf_false = random_data_false + masking_data_false + removal_data_false

    # Apply fail-safe for obfuscation using ALL the keys that can be generated
    # This ensures no matter the perturbation type, it is checked correctly.
    obf_true = remove_unperturbed_claims(
        obf_true,
        ["random_replacement_false", "masking_false", "value_removal_false"],
    )
    obf_false = remove_unperturbed_claims(
        obf_false,
        ["random_replacement_false", "masking_false", "value_removal_false"],
    )

    # Save the combined and filtered data
    save_data(obf_true, "data/perturbed_data/train_split/obfuscation_true.json")
    save_data(
        obf_false, "data/perturbed_data/train_split/obfuscation_false.json"
    )
    print("Obfuscation perturbations completed.")

    output_data_true: List[Dict[str, Any]] = range.process_json(data_true)
    output_data_false: List[Dict[str, Any]] = range.process_json(data_false)

    # Apply fail-safe for range
    output_data_true = remove_unperturbed_claims(
        output_data_true, ["range_true", "range_false"]
    )
    output_data_false = remove_unperturbed_claims(
        output_data_false, ["range_true", "range_false"]
    )

    save_data(
        output_data_true, "data/perturbed_data/train_split/range_true.json"
    )
    save_data(
        output_data_false, "data/perturbed_data/train_split/range_false.json"
    )
    print("Range perturbations completed.")


if __name__ == "__main__":
    try:
        main()
        print("All perturbation processes completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
