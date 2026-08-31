"""Utility to merge multiple JSONL files into a single JSONL file."""

import argparse
import json
import os


def merge_jsonl_files(input_files, output_file):  # noqa: C901
    """Merge multiple JSONL files into a single JSONL file.

    Args:
        input_files: List of paths to input JSONL files.
        output_file: Path to the output JSONL file.
    """
    if not input_files:
        print("Error: No input files provided.")
        return

    merged_data = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' does not exist. Skipping.")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        merged_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Failed to decode JSON in file "
                            f"'{file_path}' on line: {line.strip()}. Error: {e}"
                        )
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            continue

    if not merged_data:
        print("Error: No valid data to write to the output file.")
        return

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for record in merged_data:
                f.write(json.dumps(record) + "\n")
        print(f"Successfully merged files into '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")


def main():
    """Main function to parse arguments and merge JSONL files."""
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL files into a single JSONL file."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to input JSONL files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Path to the output JSONL file.",
    )

    args = parser.parse_args()

    merge_jsonl_files(args.input_files, args.output_file)


if __name__ == "__main__":
    main()
