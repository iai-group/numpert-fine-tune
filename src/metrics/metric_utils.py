"""Utility functions for metric calculations."""

import os
import json


def get_exclusion_list() -> set[int]:
    """Return a hardcoded list of claim_ids to exclude.

    This list includes claims with known issue of crores and lakhs.

    Returns:
        Set of claim_ids to exclude.
    """
    return set([12, 199, 215, 340, 376, 444, 504, 585, 210, 282, 609])


def flip_label(label: str) -> str:
    """Flip the boolean label from True to False or vice versa.

    Args:
        label: String representation of boolean ("True" or "False").

    Returns:
        Flipped boolean string ("False" if input was "True", "True" otherwise).
    """
    return "False" if label == "True" else "True"


def convert_bool_label_to_str(label: bool) -> str:
    """Convert boolean label to string representation.

    Args:
        label: Boolean value to convert.

    Returns:
        String representation ("True" or "False").
    """
    return str(label)


def check_file_exists_and_readable(file_path: str) -> tuple[bool, str]:
    """Check if file exists, is readable, and contains valid JSON.

    Args:
        file_path: Path to the file to check.

    Returns:
        Tuple containing success status and error message (empty if successful).
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    if os.path.getsize(file_path) == 0:
        return False, f"File is empty: {file_path}"
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                json.loads(first_line)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in file {file_path}: {e}"
    except Exception as e:
        return False, f"Error reading file {file_path}: {e}"
    return True, ""
