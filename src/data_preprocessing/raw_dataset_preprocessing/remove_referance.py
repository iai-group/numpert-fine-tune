"""Process JSON files to remove verdict-related sentences from the 'doc' field.

This module provides functions to clean evidence text, process individual JSON
files, and handle entire directories of JSON files.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Optional


def remove_gold_verdict(evidence: str) -> str:
    """Remove sentences related to the verdict or ruling from the evidence text.

    This function removes the last sentence and filters out sentences containing
    specific keywords or phrases related to verdicts.

    Args:
        evidence (str): The original evidence text.

    Returns:
        str: The processed evidence text with verdict-related sentences removed.
    """
    if not evidence:
        return evidence

    # Remove the last sentence, which might contain the verdict
    sentences = evidence.split(".")
    if len(sentences) > 1:
        evidence = ".".join(sentences[:-1])
    else:
        evidence = sentences[0]  # Only one sentence exists

    # Compile regex patterns for efficiency
    verdict_pattern = re.compile(r"(our verdict|our ruling)", re.IGNORECASE)
    filter_words_pattern = re.compile(
        r"\b(true|false|conflicting|correct|incorrect|misleading)\b",
        re.IGNORECASE,
    )

    # Split the evidence into sentences
    sentences = evidence.split(".")
    filtered_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue  # Skip empty sentences

        # If "our verdict" or "our ruling" is found, stop processing further
        if verdict_pattern.search(sentence):
            break

        # Skip sentences containing any of the filter words
        if filter_words_pattern.search(sentence):
            continue

        # Append passing sentences to the list
        filtered_sentences.append(sentence)

    # Join the filtered sentences back into a single string
    if filtered_sentences:
        return ". ".join(filtered_sentences) + "."
    else:
        return ""


def process_json_file(file_path: Path, dry_run: bool = False) -> None:
    """Process a single JSON file by applying the remove_gold_verdict function.

    Args:
        file_path (Path): Path to the JSON file.
        dry_run (bool): If True, does not write changes to the file.

    Returns:
        None
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure the data is a list
        if not isinstance(data, list):
            print(f"Skipping {file_path}: JSON data is not a list.")
            return

        modified = False  # Flag to check if any modification is done

        # Iterate over each entry and process the 'doc' field
        for entry in data:
            if "doc" in entry and isinstance(entry["doc"], str):
                original_doc = entry["doc"]
                processed_doc = remove_gold_verdict(original_doc)
                if original_doc != processed_doc:
                    entry["doc"] = processed_doc
                    modified = True
                    print(
                        f"Processed 'doc' in {file_path}:\nOriginal: "
                        f"{original_doc}\nProcessed: {processed_doc}\n"
                    )
            else:
                print(
                    f"No 'doc' field found or 'doc' is not a string in "
                    f"an entry of {file_path}."
                )

        if modified and not dry_run:
            # Write the processed data back to the file
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Updated file: {file_path}\n")
        elif modified and dry_run:
            print(f"Dry run enabled. Changes not saved for file: {file_path}\n")
        else:
            print(f"No changes made to file: {file_path}\n")

    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' does not contain valid JSON.")
    except Exception as e:
        print(
            f"An unexpected error occurred while processing '{file_path}': {e}"
        )


def find_json_files(directory: Path) -> List[Path]:
    """Recursively find all JSON files within the given directory.

    Args:
        directory (Path): The root directory to search.

    Returns:
        List[Path]: A list of Paths to JSON files.
    """
    return list(directory.rglob("*.json"))


def process_folder(  # noqa: C901
    input_dir: Path, output_dir: Optional[Path] = None, dry_run: bool = False
) -> None:
    """Process all JSON files in the input directory and its subdirectories.

    Args:
        input_dir (Path): Path to the input directory.
        output_dir (Path): Path to the output directory. If None, files are
            overwritten.
        dry_run (bool): If True, does not write changes to the files.

    Returns:
        None
    """
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: The input directory '{input_dir}' does not exist")
        return

    json_files = find_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in the directory '{input_dir}'.")
        return

    print(f"Found {len(json_files)} JSON file(s) in '{input_dir}'.\n")

    for json_file in json_files:
        if output_dir:
            # Determine the relative path and create corresponding directories
            relative_path = json_file.relative_to(input_dir)
            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Read from the original file and write to the target path
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure the data is a list
                if not isinstance(data, list):
                    print(f"Skipping {json_file}: JSON data is not a list.")
                    continue

                modified = False  # Flag to check if any modification is done

                # Iterate over each entry and process the 'doc' field
                for entry in data:
                    if "doc" in entry and isinstance(entry["doc"], str):
                        original_doc = entry["doc"]
                        processed_doc = remove_gold_verdict(original_doc)
                        if original_doc != processed_doc:
                            entry["doc"] = processed_doc
                            modified = True
                            print(
                                f"Processed 'doc' in {json_file}:\nOriginal: "
                                f"{original_doc}\nProcessed: {processed_doc}\n"
                            )
                    else:
                        print(
                            f"No 'doc' field found or 'doc' is not a string in "
                            f"an entry of {json_file}."
                        )

                if modified and not dry_run:
                    # Write the processed data to the target file
                    with target_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    print(f"Saved updated file to: {target_path}\n")
                elif modified and dry_run:
                    print(
                        f"Dry run enabled. Changes not saved for file: "
                        f"{json_file}\n"
                    )
                else:
                    print(f"No changes made to file: {json_file}\n")

            except json.JSONDecodeError:
                print(f"Error: The file '{json_file}' is invalid JSON.")
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing "
                    f"'{json_file}': {e}"
                )

        else:
            # Overwrite the original files
            process_json_file(json_file, dry_run=dry_run)


def main():
    """Main entry point for processing JSON files to remove verdict sentences.

    This function parses command-line arguments to specify the input directory,
    output directory, and whether to perform a dry run.
    """
    parser = argparse.ArgumentParser(
        description="Process JSON files to remove verdict-related sentences "
        "from the 'doc' field."
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to the input directory containing JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default=None,
        help="Path to the output directory. If not specified, original files "
        "will be overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing changes to the files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_directory).resolve()
    output_dir = (
        Path(args.output_directory).resolve() if args.output_directory else None
    )
    dry_run = args.dry_run

    if output_dir and not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return

    process_folder(input_dir, output_dir=output_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
