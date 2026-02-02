"""Script to run RoBERTa evaluations using parameter sets from a JSON file."""

import subprocess
import sys
from pathlib import Path
import logging
import json
import argparse


def setup_logging():
    """Configures logging to output to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("roberta_evaluation_runs.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_parameters(json_path):
    """Loads parameter sets from a JSON file.

    Args:
        json_path (Path): Path object pointing to the JSON file.

    Returns:
        list: List of parameter dictionaries.
    """
    logging.debug(f"Attempting to load parameters from: {json_path}")
    if not json_path.is_file():
        logging.error(f"Parameters file not found at: {json_path}")
        sys.exit(1)

    with json_path.open("r") as f:
        try:
            parameters = json.load(f)
            logging.info(
                f"Loaded {len(parameters)} parameter sets from {json_path}"
            )
            return parameters
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {json_path}: {e}")
            sys.exit(1)


def run_roberta_inference(params, script_path):  # noqa: C901
    """Runs the RoBERTa inference script with the given parameters.

    Args:
        params (dict): Dictionary containing parameters for the RoBERTa script.
        script_path (Path): Path object pointing to the RoBERTa script.
    """
    # Required arguments
    cmd = [
        sys.executable,
        str(script_path),
        params["input_file"],
        params["model_path"],
        params["output_file"],
    ]

    # Optional arguments
    if params.get("claim_key"):
        cmd.extend(["--claim_key", params["claim_key"]])
    if params.get("max_length"):
        cmd.extend(["--max_length", str(params["max_length"])])
    if params.get("device"):
        cmd.extend(["--device", params["device"]])
    if params.get("show_model_output", False):
        cmd.append("--show_model_output")
    if params.get("label_flip", False):
        cmd.append("--label_flip")

    # Log the command being executed
    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        logging.info(
            f"Success: {params['input_file']} -> {params['output_file']}"
        )
        if result.stdout:
            logging.debug(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"STDERR:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running with parameters: {params}")
        logging.error(f"Return Code: {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logging.error(f"STDERR:\n{e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def validate_parameters(parameters):
    """Validates that all required fields are present in each parameter set.

    Args:
        parameters (list): List of parameter dictionaries.

    Returns:
        bool: True if all parameter sets are valid, False otherwise.
    """
    required_fields = {"input_file", "model_path", "output_file"}

    for idx, params in enumerate(parameters, start=1):
        missing_fields = required_fields - set(params.keys())
        if missing_fields:
            logging.error(
                f"Parameter set {idx} is missing"
                "required fields: {missing_fields}"
            )
            return False

        # Validate that files exist
        input_path = Path(params["input_file"])
        if not input_path.is_file():
            logging.error(f"Input file not found: {input_path}")
            return False

    return True


def main():
    """Main function to parse arguments and run RoBERTa evaluations."""
    setup_logging()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run multiple RoBERTa evaluations with different parameter sets."  # noqa E501
    )
    parser.add_argument(
        "--parameters_file",
        type=Path,
        required=True,
        help="Path to the JSON file containing parameter sets",
    )
    parser.add_argument(
        "--script_path",
        type=Path,
        required=True,
        help="Path to the RoBERTa inference script to run",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing remaining files if an error occurs",
    )

    args = parser.parse_args()

    # Verify script exists
    if not args.script_path.is_file():
        logging.error(
            f"RoBERTa inference script not found at: {args.script_path}"
        )
        sys.exit(1)

    # Load parameter sets from the JSON file
    parameter_sets = load_parameters(args.parameters_file)

    # Validate parameters before starting
    if not validate_parameters(parameter_sets):
        logging.error("Parameter validation failed. Exiting.")
        sys.exit(1)

    logging.info(
        f"Starting RoBERTa evaluations using script at: {args.script_path}"
    )

    successful_runs = 0
    failed_runs = 0

    for idx, params in enumerate(parameter_sets, start=1):
        logging.info(f"Starting evaluation {idx}/{len(parameter_sets)}")
        logging.info(f"Parameters: {params}")

        try:
            run_roberta_inference(params, args.script_path)
            successful_runs += 1
        except Exception as e:
            failed_runs += 1
            logging.error(f"Failed to process parameter set {idx}: {e}")
            if not args.continue_on_error:
                logging.error(
                    "Stopping due to error. Use --continue_on_error"
                    " to continue processing."
                )
                break

    # Log summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Total runs attempted: {idx}")
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {failed_runs}")

    if failed_runs > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
