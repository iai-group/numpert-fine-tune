"""Main script to apply perturbations on numerical claims in a dataset."""

import argparse
from typing import Any, Dict, List

import src.data_preprocessing.perturbations_engine.approximation as approximation
import src.data_preprocessing.perturbations_engine.negative as negative
import src.data_preprocessing.perturbations_engine.numeration as numeration
import src.data_preprocessing.perturbations_engine.obfuscate as obfuscate
import src.data_preprocessing.perturbations_engine.range as range_pert
from src.data_preprocessing.perturbations_engine.utils.load_save import (
    load_data,
    save_data,
)

SPLIT = "test"
INPUT_TRUE = (
    "data/raw_data/binary_data/spacy_tagged/"
    "{split}_true_claims_tagged_normalized_numbers.json"
)
INPUT_FALSE = (
    "data/raw_data/binary_data/spacy_tagged/"
    "{split}_false_claims_tagged_normalized_numbers.json"
)
OUTPUT_DIR = "data/perturbed_data/{split}_split"


def get_paths(split: str) -> tuple[str, str, str]:
    """Resolves file paths for a given data split.

    Args:
        split: The dataset split to use. One of 'train', 'val',
          or 'test'.

    Returns:
        A tuple of (input_true_path, input_false_path, output_dir).
    """
    return (
        INPUT_TRUE.format(split=split),
        INPUT_FALSE.format(split=split),
        OUTPUT_DIR.format(split=split),
    )


def remove_unperturbed_claims(
    data: List[Dict[str, Any]],
    perturbation_keys: List[str],
) -> List[Dict[str, Any]]:
    """Removes claims where perturbations are identical to the original.

    Filters out entries where any of the specified perturbation fields
    match the normalized claim, indicating the perturbation had no
    effect.

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
        is_unperturbed = any(
            item.get(key) == normalized_claim
            for key in perturbation_keys
            if key in item
        )

        if is_unperturbed:
            removed_count += 1
        else:
            filtered_data.append(item)

    if removed_count > 0:
        print(f"Removed {removed_count} unperturbed claims.")

    return filtered_data


def save_splits(
    results: Dict[str, List[Dict[str, Any]]],
    name: str,
    perturbation_keys: List[str],
    output_dir: str,
) -> None:
    """Applies the fail-safe filter and saves true/false splits.

    Args:
        results: Dict with 'true' and 'false' splits from
          process_json.
        name: Perturbation name used in the output filename
          (e.g. 'numeration').
        perturbation_keys: Keys to check in the fail-safe filter.
        output_dir: Directory to write output files to.
    """
    for label, split_data in results.items():
        split_data = remove_unperturbed_claims(split_data, perturbation_keys)
        if split_data:
            path = f"{output_dir}/{name}_{label}.json"
            save_data(split_data, path)
            print(f"  Saved {len(split_data)} entries to: {path}")


def split_by_label(
    data: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Splits a flat list of entries into sub-lists keyed by label.

    Args:
        data: Flat list of claim dictionaries, each containing a
          'label' field.

    Returns:
        A dictionary mapping each label to its list of entries.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        label = entry.get("label", "UNKNOWN")
        results.setdefault(label, []).append(entry)
    return results


def run_perturbations(
    input_true: str,
    input_false: str,
    output_dir: str,
) -> None:
    """Runs all perturbation modules for a given split.

    Loads data from the provided paths, applies all perturbation
    types, and saves the results to the output directory.

    Args:
        input_true: Path to the JSON file containing true claims.
        input_false: Path to the JSON file containing false claims.
        output_dir: Directory where perturbed output files are saved.
    """
    data_true: List[Dict[str, Any]] = load_data(input_true)
    data_false: List[Dict[str, Any]] = load_data(input_false)
    data_all = data_true + data_false

    print("Running approximation perturbations...")
    approximation.process_claims(
        input_true,
        f"{output_dir}/approximation_True.json",
    )
    approximation.process_claims(
        input_false,
        f"{output_dir}/approximation_False.json",
    )

    print("Running negative number perturbations...")
    save_splits(
        split_by_label(negative.process_json(data_all)),
        name="negative",
        perturbation_keys=["perturb_type2"],
        output_dir=output_dir,
    )

    print("Running numeration perturbations...")
    save_splits(
        split_by_label(numeration.process_json(data_all)),
        name="numeration",
        perturbation_keys=["perturb_type1", "perturb_type2"],
        output_dir=output_dir,
    )

    print("Running range perturbations...")
    save_splits(
        split_by_label(range_pert.process_json(data_all)),
        name="range",
        perturbation_keys=["perturb_type1", "perturb_type2"],
        output_dir=output_dir,
    )

    print("Running obfuscation perturbations...")
    save_splits(
        obfuscate.process_json(data_all),
        name="obfuscation",
        perturbation_keys=[
            "perturb_type3_random",
            "perturb_type3_masked",
            "perturb_type3_removed",
        ],
        output_dir=output_dir,
    )


def main() -> None:
    """Parses arguments and runs perturbations for the selected split."""
    parser = argparse.ArgumentParser(
        description="Apply numerical perturbations to claim datasets."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default=SPLIT,
        help="Dataset split to process. Use 'all' to process all splits.",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        input_true, input_false, output_dir = get_paths(split)
        print(f"\n{'=' * 50}")
        print(f"Processing split: {split}")
        print(f"{'=' * 50}")
        run_perturbations(input_true, input_false, output_dir)


if __name__ == "__main__":
    try:
        main()
        print("\nAll perturbation processes completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
