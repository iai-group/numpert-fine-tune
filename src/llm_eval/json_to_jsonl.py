"""Takes a JSON file and converts it to a JSONL file for OpenAI Batch API."""

import json
import argparse
import sys
import os

# Add the parent directory to the Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try the original import path first
    from src.numpert_prompt.prompt import (
        get_fact_check_system_prompt,
        get_fact_check_user_prompt,
        PromptType,
    )
except ImportError:
    try:
        # If that fails, try importing from the current directory
        from src.numpert_prompt.prompt import (
            get_fact_check_system_prompt,
            get_fact_check_user_prompt,
            PromptType,
        )
    except ImportError:
        # If both fail, try importing from llm_eval in the same directory
        from src.numpert_prompt.prompt import (
            get_fact_check_system_prompt,
            get_fact_check_user_prompt,
            PromptType,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a JSON file to a JSONL file for OpenAI Batch API."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="The name of the model to use for the API requests.",
    )

    parser.add_argument(
        "--reasoning_model_name",
        type=str,
        help="The name of the reasoning model to use for the API requests.",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default=PromptType.ZERO_SHOT.value,
        choices=[pt.value for pt in PromptType],
        help=(
            "Which PromptType to use when building the user prompt. "
            f"Default: {PromptType.ZERO_SHOT.value}."
        ),
    )

    parser.add_argument(
        "--negative_examples_synthetic_all_combined",
        action="store_true",
        help="Include all combined synthetic negative examples in the prompt.",
    )

    parser.add_argument(
        "--response_format_openai",
        action="store_true",
        help="Use OpenAI structured response format.",
    )

    parser.add_argument(
        "--response_format_gemini",
        action="store_true",
        help="Use Gemini structured response format.",
    )

    parser.add_argument(
        "--input_path", required=True, help="Path to the input JSON file."
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where the output JSONL file will be saved.",
    )

    parser.add_argument(
        "--claim_keyname",
        default="claim",
        help="Key name for the claim field in the input JSON.",
    )

    parser.add_argument(
        "--dataset_type",
        default="default",
        help="Type of dataset for custom_id generation.",
    )

    parser.add_argument(
        "--consolidated",
        action="store_true",
        help="Create one consolidated JSONL file instead of separate files.",
    )

    parser.add_argument(
        "--consolidated_output",
        default="consolidated_batch.jsonl",
        help="Path for the consolidated JSONL file when using --consolidated.",
    )

    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        help="Enable reasoning for reasoning models (o1, o3, gpt-5)",
    )

    parser.add_argument(
        "--test_split_dir",
        type=str,
        default=None,
        help=(
            "Directory containing test_split JSON files. When set, ignores "
            "the built-in dataset configs and emits one JSONL per JSON file. "
            "Each entry produces one row for 'normalized_claim' plus one row "
            "for every 'perturb_type*' key present on that entry."
        ),
    )

    parser.add_argument(
        "--test_split_output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for per-file JSONLs when --test_split_dir is "
            "set. Defaults to '<test_split_dir>/jsonl/'."
        ),
    )

    return parser.parse_args()


def is_reasoning_model(model_name):
    """Check if the model is a reasoning model (o3, o1, etc.)."""
    reasoning_models = ["o1", "o3", "gpt-5"]
    return any(
        reasoning_model in model_name.lower()
        for reasoning_model in reasoning_models
    )


def is_gemini_model(model_name):
    """Check if the model is a Gemini model."""
    return "gemini" in model_name.lower()


def get_openai_response_format():
    """Get OpenAI-style response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ClassificationResponse",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "boolean",
                        "description": "True if the claim is supported by the evidence, False otherwise",  # noqa: E501
                    }
                },
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }


def get_gemini_response_format():
    """Get Gemini-style response format."""
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "boolean",
                "description": "True if the claim is supported by the evidence, False otherwise",  # noqa: E501
            }
        },
        "required": ["label"],
    }


def get_structured_prompt_for_gemini(
    claim, evidence, prompt_type: PromptType = PromptType.ZERO_SHOT
) -> str:
    """Get a more explicit prompt for Gemini structured output."""
    base_prompt = get_fact_check_user_prompt(
        claim=claim,
        evidence=evidence,
        prompt_type=prompt_type,
    )
    return f"""{base_prompt}"""


def create_consolidated_jsonl(  # noqa: C901
    datasets_config, consolidated_output_path, args
):
    """Create a single consolidated JSONL file from multiple dataset configs.

    Args:
        datasets_config (list): List of tuples (input_path, claim_keyname,
            dataset_type)
        consolidated_output_path (str): Path for the consolidated JSONL file
        args: Parsed command line arguments
    """
    print(
        f"Creating consolidated JSONL file at '{consolidated_output_path}'..."
    )

    # Determine which model to use
    model_name = (
        args.reasoning_model_name
        if args.reasoning_model_name
        else args.model_name
    )
    is_reasoning = is_reasoning_model(model_name)
    is_gemini = is_gemini_model(model_name)

    # Log reasoning configuration
    if is_reasoning:
        if args.enable_reasoning:
            print(f"Reasoning enabled for {model_name}")
        else:
            print(
                f"Reasoning disabled for {model_name} (using standard parameters)"  # noqa: E501
            )

    total_entries = 0

    try:
        with open(
            consolidated_output_path, "w", encoding="utf-8"
        ) as consolidated_file:
            for input_path, claim_keyname, dataset_type in datasets_config:
                print(
                    f"Processing {input_path} with claim key '{claim_keyname}' and dataset type '{dataset_type}'..."  # noqa: E501
                )

                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        input_data = json.load(f)
                except FileNotFoundError:
                    print(
                        f"Warning: Input file not found at '{input_path}'. Skipping."  # noqa: E50
                    )
                    continue
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Invalid JSON in file '{input_path}': {e}. Skipping."  # noqa: E50
                    )
                    continue

                if not isinstance(input_data, list):
                    print(
                        f"Warning: Input JSON in '{input_path}' is not a list. Skipping."  # noqa: E50
                    )
                    continue

                entries_processed = 0
                for i, entry in enumerate(input_data):
                    # Extract data from the input JSON object
                    custom_id = entry.get("claim_id")
                    label = entry.get("label")
                    claim = entry.get(claim_keyname)
                    evidence = entry.get("doc")

                    if not all([custom_id is not None, claim, evidence]):
                        print(
                            f"Warning: Skipping entry {i} in '{input_path}' due to missing required fields"  # noqa: E50
                        )
                        continue

                    prompt_type = PromptType(args.prompt_type)
                    if is_gemini and args.response_format_openai:
                        user_content = get_structured_prompt_for_gemini(
                            claim, evidence, prompt_type
                        )
                    else:
                        user_content = get_fact_check_user_prompt(
                            claim=claim,
                            evidence=evidence,
                            prompt_type=prompt_type,
                        )

                    # Create the body for the batch request
                    body = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": get_fact_check_system_prompt(),
                            },
                            {"role": "user", "content": user_content},
                        ],
                    }

                    # Add structured output format if requested
                    if args.response_format_openai:
                        if is_gemini:
                            body[
                                "response_format"
                            ] = get_gemini_response_format()
                        else:
                            body[
                                "response_format"
                            ] = get_openai_response_format()

                    # Configure model-specific parameters
                    if not (is_reasoning and args.enable_reasoning):
                        body["max_tokens"] = 50  # noqa: E501
                        body["temperature"] = 0.0  # noqa: E501

                    # Create the JSON object for the batch request
                    json_request = {
                        "custom_id": f"{custom_id}-{label}-{claim_keyname}-{dataset_type}",  # noqa: E501
                        "method": "POST",  # noqa: E501
                        "url": "/v1/chat/completions",  # noqa: E501
                        "body": body,
                    }

                    # Write the JSON object as a line in the JSONL file
                    consolidated_file.write(json.dumps(json_request) + "\n")
                    entries_processed += 1
                    total_entries += 1

                print(
                    f"Processed {entries_processed} entries from {input_path}"  # noqa: E501
                )

        print(
            f"Consolidated JSONL file created successfully with {total_entries} total entries."  # noqa: E501
        )
        return True

    except IOError as e:
        print(f"Error writing to consolidated output file: {e}")
        return False


def create_jsonl_from_json(  # noqa: C901
    input_path, output_path, claim_keyname, dataset_type, args=None
):
    """Create a JSONL file from a JSON file for OpenAI Batch API."""
    print(f"Reading data from '{input_path}'...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Input file not found at '{input_path}'. Please create it."
        )
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return False

    if not isinstance(input_data, list):
        print("Error: Input JSON must be a list of objects.")
        return False

    print(f"Creating JSONL file at '{output_path}'...")

    # Determine which model to use
    model_name = (
        args.reasoning_model_name
        if args.reasoning_model_name
        else args.model_name
    )
    is_reasoning = is_reasoning_model(model_name)
    is_gemini = is_gemini_model(model_name)

    # Log reasoning configuration
    if is_reasoning:
        if args.enable_reasoning:
            print(f"Reasoning enabled for {model_name}")
        else:
            print(
                f"Reasoning disabled for {model_name} (using standard parameters)"  # noqa: E501
            )

    successful_entries = 0

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for i, entry in enumerate(input_data):
                # Extract data from the input JSON object
                custom_id = entry.get("claim_id")
                label = entry.get("label")
                claim = entry.get(claim_keyname)
                evidence = entry.get("doc")

                if not all([custom_id is not None, claim, evidence]):
                    print(
                        f"Warning: Skipping entry {i} due to missing required fields: {entry}"  # noqa: E501
                    )
                    continue

                prompt_type = PromptType(args.prompt_type)
                if is_gemini and args.response_format_openai:
                    user_content = get_structured_prompt_for_gemini(
                        claim, evidence, prompt_type
                    )
                else:
                    user_content = get_fact_check_user_prompt(
                        claim=claim,
                        evidence=evidence,
                        prompt_type=prompt_type,
                    )

                body = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": get_fact_check_system_prompt(),
                        },
                        {"role": "user", "content": user_content},
                    ],
                }

                # Add structured output format if requested
                if args.response_format_openai:
                    if is_gemini:
                        body["response_format"] = get_gemini_response_format()
                    else:
                        body["response_format"] = get_openai_response_format()

                # Configure model-specific parameters
                if not (is_reasoning and args.enable_reasoning):
                    body["max_tokens"] = 50
                    body["temperature"] = 0.0

                json_request = {
                    "custom_id": f"{custom_id}-{label}-{claim_keyname}-{dataset_type}",  # noqa: E501
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }

                # Write the JSON object as a line in the JSONL file
                outfile.write(json.dumps(json_request) + "\n")
                successful_entries += 1

        print(
            f"JSONL file created successfully with {successful_entries} entries."  # noqa: E501
        )
        return True

    except IOError as e:
        print(f"Error writing to output file: {e}")
        return False


def _build_request_body(model_name, claim, evidence, args):
    """Build the chat-completions body shared across modes."""
    is_reasoning = is_reasoning_model(model_name)
    is_gemini = is_gemini_model(model_name)

    prompt_type = PromptType(args.prompt_type)
    if is_gemini and args.response_format_openai:
        user_content = get_structured_prompt_for_gemini(
            claim, evidence, prompt_type
        )
    else:
        user_content = get_fact_check_user_prompt(
            claim=claim,
            evidence=evidence,
            prompt_type=prompt_type,
        )

    body = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": get_fact_check_system_prompt(),
            },
            {"role": "user", "content": user_content},
        ],
    }

    if args.response_format_openai:
        if is_gemini:
            body["response_format"] = get_gemini_response_format()
        else:
            body["response_format"] = get_openai_response_format()

    if not (is_reasoning and args.enable_reasoning):
        body["max_tokens"] = 50
        body["temperature"] = 0.0

    return body


def create_jsonl_from_test_split(input_path, output_path, dataset_type, args):
    """Emit one JSONL line per claim variant for a test_split JSON file.

    Each entry yields:
      - one row for 'normalized_claim'
      - one row for every key matching 'perturb_type*' on that entry
    The variant key is recorded inside the custom_id so the source claim is
    recoverable downstream.
    """
    print(f"Reading data from '{input_path}'...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return False

    if not isinstance(input_data, list):
        print("Error: Input JSON must be a list of objects.")
        return False

    print(f"Creating JSONL file at '{output_path}'...")

    model_name = (
        args.reasoning_model_name
        if args.reasoning_model_name
        else args.model_name
    )
    is_reasoning = is_reasoning_model(model_name)
    if is_reasoning:
        state = "enabled" if args.enable_reasoning else "disabled"
        print(f"Reasoning {state} for {model_name}")

    written = 0
    skipped = 0

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for i, entry in enumerate(input_data):
                custom_id = entry.get("claim_id")
                label = entry.get("label")
                evidence = entry.get("doc")
                normalized_claim = entry.get("normalized_claim")

                if not all([custom_id is not None, evidence, normalized_claim]):
                    print(
                        f"Warning: Skipping entry {i} in '{input_path}' due "
                        "to missing claim_id/doc/normalized_claim."
                    )
                    skipped += 1
                    continue

                variants = [("normalized_claim", normalized_claim)]
                for key, value in entry.items():
                    if key.startswith("perturb_type") and isinstance(
                        value, str
                    ):
                        variants.append((key, value))

                for variant_key, claim in variants:
                    if not claim:
                        skipped += 1
                        continue
                    body = _build_request_body(
                        model_name, claim, evidence, args
                    )
                    json_request = {
                        "custom_id": (
                            f"{custom_id}-{label}-{variant_key}-{dataset_type}"
                        ),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                    outfile.write(json.dumps(json_request) + "\n")
                    written += 1

        print(
            f"Wrote {written} rows to '{output_path}' "
            f"({skipped} variants skipped)."
        )
        return True
    except IOError as e:
        print(f"Error writing to output file: {e}")
        return False


def run_test_split_mode(args):
    """Process every JSON file in args.test_split_dir."""
    test_dir = args.test_split_dir
    if not os.path.isdir(test_dir):
        print(f"Error: --test_split_dir '{test_dir}' is not a directory.")
        return False

    output_dir = args.test_split_output_dir or os.path.join(test_dir, "jsonl")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing per-file JSONLs to '{output_dir}'.")

    json_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".json"))
    if not json_files:
        print(f"No .json files found in '{test_dir}'.")
        return False

    overall_ok = True
    for filename in json_files:
        input_path = os.path.join(test_dir, filename)
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{stem}.jsonl")
        ok = create_jsonl_from_test_split(
            input_path=input_path,
            output_path=output_path,
            dataset_type=stem,
            args=args,
        )
        if not ok:
            overall_ok = False
    return overall_ok


def main():
    """Main function to run the script."""
    args = parse_args()

    if args.test_split_dir:
        success = run_test_split_mode(args)
        if not success:
            exit(1)
        return

    # Define all dataset configurations
    datasets_config = [
        # approximation data
        (
            "data/perturbed_data/false_dataset/jsons/approximation_false.json",
            "approximation_true",
            "ft2f",
        ),
        (
            "data/perturbed_data/false_dataset/jsons/approximation_false.json",
            "approximation_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/approximation_true.json",
            "approximation_true",
            "t2t",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/approximation_true.json",
            "approximation_false",
            "t2f",
        ),
        # negative data
        (
            "data/perturbed_data/false_dataset/jsons/negative_false.json",
            "negative_number_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/negative_true.json",
            "negative_number_false",
            "t2f",
        ),
        # numeration data
        (
            "data/perturbed_data/false_dataset/jsons/numeration_false.json",
            "numeration_true",
            "ft2f",
        ),
        (
            "data/perturbed_data/false_dataset/jsons/numeration_false.json",
            "numeration_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/numeration_true.json",
            "numeration_true",
            "t2t",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/numeration_true.json",
            "numeration_false",
            "t2f",
        ),
        # obfuscation
        (
            "data/perturbed_data/false_dataset/jsons/obfuscation_false.json",
            "random_replacement_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/obfuscation_true.json",
            "random_replacement_false",
            "t2f",
        ),
        (
            "data/perturbed_data/false_dataset/jsons/obfuscation_false.json",
            "masking_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/obfuscation_true.json",
            "masking_false",
            "t2f",
        ),
        # range perturbation
        (
            "data/perturbed_data/false_dataset/jsons/range_false.json",
            "range_true",
            "ft2f",
        ),
        (
            "data/perturbed_data/false_dataset/jsons/range_false.json",
            "range_false",
            "ff2f",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/range_true.json",
            "range_true",
            "t2t",
        ),
        (
            "data/perturbed_data/true_dataset/jsons/range_true.json",
            "range_false",
            "t2f",
        ),
    ]

    if args.consolidated:
        # Create one consolidated JSONL file
        success = create_consolidated_jsonl(
            datasets_config, args.consolidated_output, args
        )
    else:
        # Create individual JSONL files (original behavior)
        success = True
        output_configs = [
            # approximation data
            (
                "data/perturbed_data/false_dataset/jsons/approximation_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/approximation_false_dataset_ft2f.jsonl",  # noqa: E501
                "approximation_true",
                "ft2f",
            ),
            (
                "data/perturbed_data/false_dataset/jsons/approximation_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/approximation_false_dataset_ff2f.jsonl",  # noqa: E501
                "approximation_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/approximation_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/approximation_true_dataset_t2t.jsonl",  # noqa: E501
                "approximation_true",
                "t2t",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/approximation_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/approximation_true_dataset_t2f.jsonl",  # noqa: E501
                "approximation_false",
                "t2f",
            ),
            # negative data
            (
                "data/perturbed_data/false_dataset/jsons/negative_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/negative_false_dataset_ff2f.jsonl",  # noqa: E501
                "negative_number_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/negative_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/negative_true_dataset_t2f.jsonl",  # noqa: E501
                "negative_number_false",
                "t2f",
            ),
            # numeration data
            (
                "data/perturbed_data/false_dataset/jsons/numeration_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/numeration_false_dataset_ft2f.jsonl",  # noqa: E501
                "numeration_true",
                "ft2f",
            ),
            (
                "data/perturbed_data/false_dataset/jsons/numeration_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/numeration_false_dataset_ff2f.jsonl",  # noqa: E501
                "numeration_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/numeration_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/numeration_true_dataset_t2t.jsonl",  # noqa: E501
                "numeration_true",
                "t2t",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/numeration_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/numeration_true_dataset_t2f.jsonl",  # noqa: E501
                "numeration_false",
                "t2f",
            ),
            # obfuscation
            (
                "data/perturbed_data/false_dataset/jsons/obfuscation_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/obfuscation_false_dataset_ff2f.jsonl",  # noqa: E501
                "random_replacement_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/obfuscation_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/obfuscation_true_dataset_t2f.jsonl",  # noqa: E501
                "random_replacement_false",
                "t2f",
            ),
            (
                "data/perturbed_data/false_dataset/jsons/obfuscation_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/masking_false_dataset_ff2f.jsonl",  # noqa: E501
                "masking_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/obfuscation_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/masking_true_dataset_t2f.jsonl",  # noqa: E501
                "masking_false",
                "t2f",
            ),
            # range perturbation
            (
                "data/perturbed_data/false_dataset/jsons/range_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/range_false_dataset_ft2f.jsonl",  # noqa: E501
                "range_true",
                "ft2f",
            ),
            (
                "data/perturbed_data/false_dataset/jsons/range_false.json",  # noqa: E501
                "data/perturbed_data/false_dataset/range_false_dataset_ff2f.jsonl",  # noqa: E501
                "range_false",
                "ff2f",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/range_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/range_true_dataset_t2t.jsonl",  # noqa: E501
                "range_true",
                "t2t",
            ),
            (
                "data/perturbed_data/true_dataset/jsons/range_true.json",  # noqa: E501
                "data/perturbed_data/true_dataset/range_true_dataset_t2f.jsonl",  # noqa: E501
                "range_false",
                "t2f",
            ),
        ]

        for (
            input_path,
            output_path,
            claim_keyname,
            dataset_type,
        ) in output_configs:
            file_success = create_jsonl_from_json(
                input_path, output_path, claim_keyname, dataset_type, args
            )
            if not file_success:
                success = False

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
