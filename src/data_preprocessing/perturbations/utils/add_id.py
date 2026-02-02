"""A utility to add unique IDs to JSON objects."""

import json
import argparse
import sys
from collections import OrderedDict


def add_ids_to_json(input_file: str, output_file: str) -> None:
    """Adds a unique 'ID' field to each JSON object.

    The JSON file is loaded and written back after assigning unique 'ID'.

    This version attempts to preserve the original order of keys in each object
    and prevents the conversion of special characters to unicode escape
    sequences.

    Args:
        input_file: The path to the input JSON file.
        output_file: The path to the output JSON file.
    """
    try:
        # Open and read the input JSON file, specifying UTF-8 encoding
        with open(input_file, "r", encoding="utf-8") as f:
            # Use object_pairs_hook=OrderedDict to load JSON objects into an
            # ordered dictionary. This preserves the original key order.
            data = json.load(f, object_pairs_hook=OrderedDict)

        # Ensure the JSON data is a list
        if not isinstance(data, list):
            print(
                "Error: The root of the JSON file must be an array/list of "
                "objects."
            )
            sys.exit(1)

        # Iterate through the list of objects using enumerate to get both the
        # index and the item. We use start=1 to begin our IDs from 1 instead
        # of 0.
        for i, item in enumerate(data, start=1):
            # Add the 'ID' field.
            item["claim_id"] = i

        # Open the output file in write mode ('w'), specifying UTF-8 encoding
        with open(output_file, "w", encoding="utf-8") as f:
            # json.dump() writes the Python object to the JSON file.
            # By using an OrderedDict, the key order from the original is
            # maintained. The 'indent=4' argument is kept to ensure the output
            # is human-readable. ensure_ascii=False tells the writer to output
            # special characters directly.
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(
            f"Successfully added IDs and saved the new file to: {output_file}"
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from the file '{input_file}'. "
            "Please check its format."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Adds a unique, iterative 'ID' field to each element in a JSON "
            "array."
        )
    )
    parser.add_argument("input_file", help="The path to the input JSON file.")
    parser.add_argument(
        "output_file", help="The path for the resulting output JSON file."
    )

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the provided file paths
    add_ids_to_json(args.input_file, args.output_file)
