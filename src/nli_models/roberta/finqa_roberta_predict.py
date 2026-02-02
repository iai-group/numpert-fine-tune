"""RoBERTa Inference Script that processes a single input JSON file.

It allows the user to specify which key contains the 'claim' and optionally flip
all labels to "False".
"""

import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[RobertaForSequenceClassification, RobertaTokenizer]:
    """Load a RoBERTa model and tokenizer.

    Args:
        model_path: Path or model identifier to load (e.g., "roberta-base" or
            a local directory).
        device: Device to load the model on ("cuda" or "cpu").

    Returns:
        The loaded model and tokenizer.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    ).to(device)
    model.eval()
    return model, tokenizer


def run_inference(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizer,
    input_text: str,
    max_length: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_model_output: bool = False,
) -> str:
    """Run inference on a single input text using RoBERTa model.

    Args:
        model (RobertaForSequenceClassification): The RoBERTa model.
        tokenizer (RobertaTokenizer): The RoBERTa tokenizer.
        input_text (str): The input string for inference.
        max_length (int): Max token length for truncation.
        device (str): "cpu" or "cuda".
        show_model_output (bool): If True, print the raw model output.

    Returns:
        str: The final prediction ("True" or "False").
    """
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    if show_model_output:
        print(f"Model logits: {logits}")
        print(f"Probabilities: {probabilities}")
        print(f"Raw prediction: {prediction}")

    # Convert numeric prediction to string (assuming 1 = True, 0 = False)
    return "True" if prediction == 1 else "False"


def inference_on_json_file(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizer,
    json_path: str,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_model_output: bool = False,
    claim_key: str = "claim",
    label_flip: bool = False,
) -> List[Dict[str, Any]]:
    """Runs inference for data and return results in the specified format.

    Each item in the JSON is expected to have 'url',
    'label', and a claim field (via `claim_key`). We add 'evaluation' based on
    model output, and use 'placeholder.' for 'evidence'. If `label_flip` is
    True, all gold labels are replaced with "False".

    Args:
        model: A RoBERTa model instance.
        tokenizer: The corresponding RoBERTa tokenizer.
        json_path: Path to a JSON file.
        max_length: Max token length. Defaults to 512.
        device: "cpu" or "cuda". Defaults to auto-detect GPU.
        show_model_output: If True, print model outputs during inference.
        claim_key: The key in the JSON file where the claim is stored
            (default: "claim").
        label_flip: If True, all gold labels are replaced with "False".

    Returns:
        List[Dict[str, Any]]: A list of results, each containing:
            "url", "claim", "evidence", "label",
            "status", "evaluation", "raw_response"
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        url = item.get("url", "")
        claim = item.get(claim_key, "")
        label = item.get("label", "")

        # Flip the label to "False" if requested
        if label_flip:
            label = "False"

        evidence = "placeholder."  # Hard-coded evidence

        # Prepare RoBERTa input
        input_text = f"{claim} </s></s> {evidence}"
        if len(input_text) > max_length:
            input_text = input_text[:max_length]

        prediction = run_inference(
            model,
            tokenizer,
            input_text,
            max_length,
            device=device,
            show_model_output=show_model_output,
        )

        # Build output dictionary
        result_item = {
            "url": url,
            "claim": claim,
            "evidence": evidence,
            "label": label,
            "status": "success",
            "evaluation": prediction,
            "raw_response": prediction,
        }

        results.append(result_item)
    return results


def save_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save a list of Python dictionaries to a JSON file.

    Args:
        data (List[Dict[str, Any]]): The data to save.
        output_path (str): Where to save the JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main(
    input_file: str,
    model_path: str,
    output_file: str,
    claim_key: str = "claim",
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_model_output: bool = False,
    label_flip: bool = False,
) -> None:
    """Runs RoBERTa inference on a single JSON file and saves results.

    Args:
        input_file: Path to the input JSON file.
        model_path: Path or model identifier for loading a RoBERTa model.
        output_file: Path to save the output JSON file.
        claim_key: The JSON key for the claim field (default: "claim").
        max_length: Max token length for RoBERTa inputs. Defaults to 512.
        device: Device ("cpu" or "cuda"). Defaults to auto.
        show_model_output: If True, print model outputs during inference.
        label_flip: If True, gold labels in the JSON are replaced with "False".
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)

    # Process the specified JSON file
    print(f"\nProcessing file: {input_file}")
    results = inference_on_json_file(
        model,
        tokenizer,
        input_file,
        max_length=max_length,
        device=device,
        show_model_output=show_model_output,
        claim_key=claim_key,
        label_flip=label_flip,
    )
    save_json(results, output_file)
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RoBERTa inference on a single JSON file to predict True/False, "  # noqa: E501
        "specifying which JSON key is used for the claim, and optionally flipping labels to 'False'."  # noqa: E501
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSON file."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path or model identifier for the RoBERTa model.",
    )
    parser.add_argument(
        "output_file", type=str, help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--claim_key",
        type=str,
        default="claim",
        help="JSON key where the claim is stored (default: 'claim').",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for RoBERTa inputs (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: 'cuda' if available else 'cpu').",  # noqa: E501
    )
    parser.add_argument(
        "--show_model_output",
        action="store_true",
        help="If set, prints the raw model output during inference.",
    )
    parser.add_argument(
        "--label_flip",
        action="store_true",
        help="If set, all gold labels in the input JSON are replaced with 'False'.",  # noqa: E501
    )

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        model_path=args.model_path,
        output_file=args.output_file,
        claim_key=args.claim_key,
        max_length=args.max_length,
        device=args.device,
        show_model_output=args.show_model_output,
        label_flip=args.label_flip,
    )
