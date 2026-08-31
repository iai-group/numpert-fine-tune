import json

from anyio import Path

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "averitec"
DEV_NUMERICAL = DATA_ROOT / "averitec_dev_numerical.json"
TRAIN_NUMERICAL = DATA_ROOT / "averitec_train_numerical.json"
PERTURBED_DEV = DATA_ROOT / "perturbed" / "dev_split"

def filter_label(label="Conflicting", split="dev"):
    """Filters the dev or train numerical file to only include records with the specified label.

    Args:
        label: The label to filter by (default is "Conflicting").
        split: The data split to filter ("dev" or "train").
    """
    if split == "dev":
        file = DEV_NUMERICAL
    elif split == "train":
        file = TRAIN_NUMERICAL
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'dev' or 'train'.")

    with open(file) as f:
        records = json.load(f)

    filtered_records = [r for r in records if r.get("label") == label]

    output_file = DATA_ROOT / f"averitec_{split}_{label.lower()}.json"
    with open(output_file, "w") as f:
        json.dump(filtered_records, f, indent=2, ensure_ascii=False)

    print(f"Filtered {len(filtered_records)} records with label '{label}' from {file} and saved to {output_file}.")
    
if __name__ == "__main__":
    filter_label(label="Conflicting", split="dev")
    # filter_label(label="Conflicting", split="train")