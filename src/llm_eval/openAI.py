"""OpenAI evaluation script for claims based on evidence."""

import json
import argparse
import os
from typing import Any, Dict, Optional, Union
from openai import OpenAI
from time import sleep


def get_nested(
    item: Dict[str, Any], key_path: str, default: Optional[Any] = None
) -> Any:
    """Retrieve a value from a nested dictionary using a dot-separated key path.

    Args:
        item: The dictionary to search.
        key_path: Dot-separated path to the desired key.
        default: Default value to return if the path is not found.

    Returns:
        The value found at the specified path or the default.
    """
    keys = key_path.split(".")
    for key in keys:
        if isinstance(item, dict) and key in item:
            item = item[key]
        else:
            return default
    return item


def evaluate_input(
    client: OpenAI,
    claim: str,
    doc: str,
    system_message: str,
    model_name: str,
    show_model_output: bool = False,
    max_attempts: int = 5,
) -> Dict[str, Union[str, Any]]:
    """Evaluate whether a claim is True or False based on provided evidence.

    Args:
        client: OpenAI client instance.
        claim: The claim to evaluate.
        doc: The evidence supporting or refuting the claim.
        system_message: The system prompt for the model.
        model_name: The name of the GPT model to use.
        show_model_output: If True, prints the model's output.
        max_attempts: Maximum number of attempts for evaluation.

    Returns:
        A dictionary containing evaluation status and results.
    """
    function = {
        "name": "evaluate_claim",
        "description": "Evaluates if a given claim is True or False according to the provided evidence.",  # noqa E501
        "parameters": {
            "type": "object",
            "properties": {
                "evaluation": {
                    "type": "string",
                    "enum": ["True", "False"],
                    "description": "The evaluation of the claim.",
                }
            },
            "required": ["evaluation"],
            "additionalProperties": False,
        },
    }

    for attempt in range(max_attempts):
        try:
            input_text = f"[Claim]: {claim}\n[Evidence]: {doc}"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_text},
                ],
                functions=[function],
                function_call={"name": "evaluate_claim"},
                max_tokens=50,
                n=1,
                temperature=0,
            )

            message = response.choices[0].message
            if hasattr(message, "function_call") and message.function_call:
                function_call = message.function_call
                arguments = function_call.arguments
                if arguments:
                    args = json.loads(arguments)
                    evaluation = args.get("evaluation")
                    if evaluation in ["True", "False"]:
                        return {
                            "status": "success",
                            "category": evaluation,
                            "raw_response": arguments,
                        }
                    else:
                        print(
                            f"Attempt {attempt + 1}: Invalid evaluation value: {evaluation}"  # noqa E501
                        )
                else:
                    print(
                        f"Attempt {attempt + 1}: No arguments in function call"
                    )
            else:
                print(
                    f"Attempt {attempt + 1}: Assistant did not call the function"  # noqa E501
                )
                if show_model_output:
                    print(f"Model Output: {message.content}")
            sleep(1)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_attempts - 1:
                sleep(1)
                continue
            return {
                "status": "error",
                "category": "ERROR",
                "raw_response": str(e),
            }

    return {
        "status": "failed_all_attempts",
        "category": "INVALID",
        "raw_response": "No valid True/False response after maximum attempts",  # noqa E501
    }


def main(  # noqa: C901
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: Optional[str] = None,
    num_elements: Optional[int] = None,
    system_message: Optional[str] = None,
    show_model_output: bool = False,
    claim_key: str = "claim",
    label_flip: Optional[str] = None,
) -> None:
    """Process the input JSON, evaluate claims, and save the results.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        model_name: The name of the GPT model to use.
        api_key: OpenAI API key.
        num_elements: Number of elements to evaluate.
        system_message: System message for the model.
        show_model_output: If True, prints the model's output.
        claim_key: Dot-separated path to the claim field in the input JSON.
        label_flip: If set, overrides all gold labels with "True" or "False".
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Either:\n"
                "1. Set the OPENAI_API_KEY environment variable, or\n"
                "2. Pass the API key using the --api_key argument."
            )

    client = OpenAI(api_key=api_key)

    try:
        with open(input_file, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in input file: {input_file}")

    if system_message is None:
        print("Zero shot evaluation selected.")
        system_message = (
            "You are an expert evaluator. Evaluate if a given claim is True or "
            "False according to the provided evidence. Use the function "
            "`evaluate_claim` to provide your evaluation."
        )
    elif system_message == "few_shot":
        print("Few-shot evaluation selected.")
        system_message = (
            "You are an expert evaluator. Evaluate if a given claim is True or "
            "False according to the provided evidence. You must respond ONLY "
            "with [True] or [False] - no other text or explanation."
        )

    if num_elements is not None:
        data = data[:num_elements]

    results = []
    total_items = len(data)
    successful_evaluations = 0
    true_predictions = 0
    false_predictions = 0
    error_predictions = 0
    invalid_predictions = 0

    for idx, item in enumerate(data, 1):
        claim = get_nested(item, claim_key, "")
        doc = item.get("doc", "")
        url = item.get("url", "")

        label = label_flip if label_flip is not None else item.get("label", "")

        print(f"\nProcessing item {idx}/{total_items}")
        print(f"Claim: {claim}")

        evaluation = evaluate_input(
            client, claim, doc, system_message, model_name, show_model_output
        )

        result_entry = {
            "url": url,
            "claim": claim,
            "evidence": doc,
            "label": label,
            "status": evaluation.get("status"),
            "evaluation": evaluation.get("category"),
            "raw_response": evaluation.get("raw_response"),
        }

        if evaluation.get("status") == "success":
            successful_evaluations += 1
            if evaluation.get("category") == "True":
                true_predictions += 1
            elif evaluation.get("category") == "False":
                false_predictions += 1
        elif evaluation.get("status") == "error":
            error_predictions += 1
        elif evaluation.get("status") == "failed_all_attempts":
            invalid_predictions += 1

        results.append(result_entry)
        print(
            f"Evaluation: {result_entry.get('evaluation')} (Gold Label: {label})"  # noqa E501
        )

    try:
        with open(output_file, "w") as result_file:
            json.dump(results, result_file, indent=4)
    except IOError:
        raise IOError(f"Error writing to output file: {output_file}")

    print(f"\nEvaluation completed and saved to {output_file}")

    accuracy = (
        (true_predictions + false_predictions) / successful_evaluations * 100
        if successful_evaluations
        else 0
    )

    print("\nSummary:")
    print(f"Total processed: {len(results)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f" - True predictions: {true_predictions}")
    print(f" - False predictions: {false_predictions}")
    print(f"Errors encountered: {error_predictions}")
    print(f"Invalid responses after all attempts: {invalid_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate claims based on evidence using GPT."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument(
        "--model_name",
        help="Name of the GPT model to use (optional)",
        default="gpt-4o-mini",
    )
    parser.add_argument("--api_key", help="OpenAI API key")
    parser.add_argument(
        "--num_elements",
        type=int,
        help="Number of elements to evaluate (optional)",
    )
    parser.add_argument(
        "--system_message",
        help="Set system message to 'few_shot' for few-shot learning (optional) or none for default message zero-shot evaluation",  # noqa E501
    )
    parser.add_argument(
        "--show_model_output",
        action="store_true",
        help="Print the model's output for each evaluation (optional)",
    )
    parser.add_argument(
        "--claim_key",
        help="Dot-separated path to the claim field in the input JSON (default: 'claim')",  # noqa E501
        default="claim",
    )
    parser.add_argument(
        "--label_flip",
        choices=["True", "False"],
        help="Set all gold labels to 'True' or 'False', overriding original labels",  # noqa E501
        default=None,
    )

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        api_key=args.api_key,
        num_elements=args.num_elements,
        system_message=args.system_message,
        show_model_output=args.show_model_output,
        claim_key=args.claim_key,
        label_flip=args.label_flip,
    )
