"""Script to process a JSONL file containing records with 'custom_id' fields."""

import json
import argparse
from collections import Counter, defaultdict

# Hardcoded excluded IDs
EXCLUDED_IDS = {12, 199, 215, 340, 376, 444, 504, 585, 210, 282, 609}


def process_jsonl(jsonl_path):
    """Process the JSONL file and compute required metrics."""
    category_counts = Counter()
    digit_counts = Counter()
    category_by_digit = defaultdict(Counter)
    total_records = 0
    excluded_records = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                if not custom_id:
                    continue

                # Example: "1-False-approximation_true-ft2f"
                parts = custom_id.split("-", 1)
                if len(parts) != 2:
                    continue

                digit_id_str, category = parts
                try:
                    digit_id = int(digit_id_str)
                except ValueError:
                    continue

                total_records += 1
                if digit_id in EXCLUDED_IDS:
                    excluded_records += 1
                    continue

                category_counts[category] += 1
                digit_counts[digit_id] += 1
                category_by_digit[digit_id][category] += 1

            except json.JSONDecodeError:
                continue

    results = {
        "total_records": total_records,
        "excluded_records": excluded_records,
        "included_records": total_records - excluded_records,
        "category_counts": dict(category_counts),
        "digit_counts": dict(digit_counts),
        "category_by_digit": {
            str(k): dict(v) for k, v in category_by_digit.items()
        },
    }
    return results


def main():
    """Main function to parse arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process a JSONL file of custom_ids."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to JSONL file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output JSON file."
    )

    args = parser.parse_args()

    results = process_jsonl(args.input)

    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"✅ Results written to {args.output}")


if __name__ == "__main__":
    main()
