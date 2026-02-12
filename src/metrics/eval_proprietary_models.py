"""Module to evaluate propreitary models."""

import os
from typing import Any, Dict, List, Tuple
import json
from sklearn.metrics import accuracy_score

from code.metrics.eval_metrics_latex import DeltaEvalTable
from code.metrics.metric_utils import get_exclusion_list


def read_pred_file(file_name: str) -> List[Dict[str, Any]]:
    """Read prediction file.

    Args:
        file_name (str): Path to the prediction file.

    Returns:
        List[Dict[str, Any]]: List of predictions.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]
    return preds


def get_true_pred_labels(row: Dict[str, Any]) -> Tuple[str, str]:
    """Get true and predicted labels from a row.

    Args:
        row (Dict[str, Any]): A row from the dataset.

    Returns:
        Tuple[str, str]: True label and predicted label.
    """
    true_label = get_ground_truth_label(row.get("custom_id", ""))
    response = row.get("response", {})
    if response is None and row.get("error", {}).get("type", "") == "api_error":
        return true_label, ""
    body = response.get("body", {})
    choices = body.get("choices", [{}])
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", "")
    if not content:
        return true_label, ""
    pred_label = parse_pred_label(content)
    assert true_label in ["True", "False"]
    return true_label, pred_label


def get_ground_truth_label(custom_id: str) -> str:
    """Get ground truth label from a name.

    Args:
        name (str): The name of the instance.

    Returns:
        str: The ground truth label.
    """
    custom_id_parts = custom_id.split("-")
    # if it is a True -> False perturbation, return "False".
    if custom_id_parts[-1] == "t2f":
        return "False"
    return custom_id_parts[1]


def parse_pred_label(content: str) -> str:
    """Parse the predicted label from the content string.

    Args:
        content: The content string.

    Returns:
        The predicted label.
    """
    if not content:
        return ""
    try:
        return str(json.loads(clean_content(content)).get("label", ""))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return ""
    except Exception as e:
        print(clean_content(content))
        print(f"Unexpected error: {e}")
        return ""


def clean_content(content: str) -> str:
    """Clean the content string.

    Args:
        content: The content string.

    Returns:
        The cleaned content string.
    """
    return (
        content.replace("\t", "")
        .replace("\n", "")
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


def get_file_name(folder: str, model: str, perturb: str, shot: str) -> str:
    """Get the file name for the experiment results.

    Args:
        folder: The folder containing the results.
        model: The name of the model.
        perturb: The type of perturbation.
        shot: The shot type (few or one).

    Returns:
        The file name.
    """
    return f"results/{folder}/{shot}_shot/{perturb}_{shot}_{model}.jsonl"


def is_correct_experiment(
    row: Dict[str, Any],
    original_label: str,
    perturb_type: str = None,
    perturb_label: str = None,
    exclusion_list: set[int] = get_exclusion_list(),
) -> bool:
    """Check if the experiment is correct based on the row and original label.

    Args:
        row: The row from the dataset.
        original_label: The original label.

    Returns:
        True if the experiment is correct, False otherwise.
    """
    custom_id = row.get("custom_id", "")
    custom_id_parts = custom_id.split("-")
    claim_id = int(custom_id_parts[0])
    if claim_id in exclusion_list:
        return False
    pert_type_label = custom_id_parts[2]
    gt_label = custom_id_parts[1]
    if (
        pert_type_label == "normalized_claim"
        or pert_type_label == f"{perturb_type}_{perturb_label}"
    ):
        if original_label == gt_label:
            return True
    return False


def eval_experiment(
    folder: str,
    model: str,
    perturb: str,
    shot: str,
    original_label: str,
    perturb_type: str = None,
    perturb_label: str = None,
) -> Tuple[List[str], List[str], float, int]:
    """Evaluate an experiment.

    Args:
        folder: The folder containing the results.
        model: The name of the model.
        perturb: The type of perturbation.
        shot: The shot type (few or one).
        original_label: The original label.
        perturb_type: The type of perturbation.
        perturb_label: The label of the perturbation.

    Returns:
        Evaluation results containing true labels, predicted labels, accuracy,
        and invalid count.
    """
    file_name = get_file_name(folder, model, perturb, shot)
    if os.path.exists(file_name) is False:
        return [], [], 0.0, 0
    preds = read_pred_file(file_name)

    true_labels = []
    pred_labels = []
    invalid = 0
    exclusion_list = get_exclusion_list()
    for row in preds:
        if not is_correct_experiment(
            row=row,
            original_label=original_label,
            perturb_type=perturb_type,
            perturb_label=perturb_label,
            exclusion_list=exclusion_list,
        ):
            continue
        true_label, pred_label = get_true_pred_labels(row)
        if pred_label == "":
            invalid += 1
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    # print(len(true_labels), len(pred_labels), invalid)
    return (
        true_labels,
        pred_labels,
        accuracy_score(true_labels, pred_labels),
        invalid,
    )


def get_configuration() -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, str],
    List[str],
    Dict[str, str],
    List[str],
    List[str],
    Dict[str, str],
]:
    """Get configuration for perturbation evaluation.

    Returns:
        Tuple containing perturbations, models, short perturbation names,
        shots, and model display names.
    """
    perturbations = [
        "normalized_claim",
        "approximation",
        "negative_number",
        "numeration",
        "random_replacement",
        "range",
        "masking",
    ]

    models = {
        "gemini_2.5-flash": ["gemini-2-5_NO_REASONING", "gemini-2-5"],
        "gpt_o3": ["o3"],
        "gpt-4o": ["4o"],
        "gpt-4o-mini": ["4o-mini"],
        "gpt-5": ["gpt-5"],
    }

    model_display_names = {
        "gemini-2-5_NO_REASONING": "Gemini 2.5F",
        "gemini-2-5": "Gemini 2.5F (T)",
        "o3": "GPT-o3",
        "4o": "GPT-4o",
        "4o-mini": "GPT-4o-Mini",
        "gpt-5": "GPT-5",
    }

    short_pert_names = {
        # "normalized_claim": "original",
        "approximation": "approx",
        "negative_number": "neg_num",
        "numeration": "num",
        "random_replacement": "rand_repl",
        "range": "range",
        "masking": "mask",
    }

    shots_display_names = {
        "one": "0-S",
        "few": "2-S",
        "negative": "Perturbed Examples",
    }

    shots = ["few", "one", "negative"]
    # shots = ["negative"]
    original_labels = ["True", "False"]
    perturb_labels = ["false", "true"]
    # original_labels = ["True"]
    # perturb_labels = ["true"]

    return (
        perturbations,
        models,
        short_pert_names,
        shots,
        model_display_names,
        original_labels,
        perturb_labels,
        shots_display_names,
    )


if __name__ == "__main__":
    (
        perturbations,
        models,
        short_pert_names,
        shots,
        model_display_names,
        original_labels,
        perturb_labels,
        shot_display_names,
    ) = get_configuration()
    with open("results/prop_results/errors.txt", "w") as error_file:
        error_file.write(
            "folder\tmodel\tshot\tperturb_type\toriginal_label -> perturb_label\tlen(true_labels)\tinvalid\n"  # noqa: E501
        )
        for original_label in original_labels:
            for perturb_label in perturb_labels:
                table = DeltaEvalTable(metrics=("accuracy",), digits=2)
                for folder in models.keys():
                    for model in models[folder]:
                        for shot in shots:
                            (
                                true_labels,
                                pred_labels,
                                accuracy,
                                invalid,
                            ) = eval_experiment(
                                folder=folder,
                                model=model,
                                perturb="baseline",
                                shot=shot,
                                original_label=original_label,
                            )
                            if len(true_labels) == 0 or len(pred_labels) == 0:
                                continue
                            table.add_result(
                                model_display_names[model]
                                + "_"
                                + shot_display_names[shot],
                                "original",
                                y_true=true_labels,
                                y_pred=pred_labels,
                            )
                            print(
                                f"{model}\t{shot}\toriginal",
                                len(true_labels),
                                len(pred_labels),
                            )
                            for perturb_type in perturbations:
                                (
                                    true_labels,
                                    pred_labels,
                                    accuracy,
                                    invalid,
                                ) = eval_experiment(
                                    folder=folder,
                                    model=model,
                                    perturb="perturb",
                                    shot=shot,
                                    original_label=original_label,
                                    perturb_type=perturb_type,
                                    perturb_label=perturb_label,
                                )
                                if (
                                    len(true_labels) == 0
                                    or len(pred_labels) == 0
                                ):
                                    continue
                                assert len(true_labels) == len(pred_labels)
                                print(
                                    f"{model}\t{shot}\t{perturb_type}",
                                    len(true_labels),
                                    len(pred_labels),
                                )
                                table.add_result(
                                    model_display_names[model]
                                    + "_"
                                    + shot_display_names[shot],
                                    short_pert_names[
                                        perturb_type
                                    ],  # noqa: E501
                                    y_true=true_labels,
                                    y_pred=pred_labels,
                                )
                                # print(table._store)
                                error_file.write(
                                    f"{folder}\t{model}\t{shot}\t{perturb_type}\t{original_label} -> {perturb_label}\t{len(true_labels)}\t{invalid}\n"  # noqa: E501
                                )
                with open(
                    f"results/prop_results/latex_table_{original_label}_{perturb_label}.tex",  # noqa: E501
                    "w",
                ) as f:
                    f.write(
                        table.to_latex(
                            variations=list(short_pert_names.values()),
                            section_labels={
                                "0-S": [
                                    model
                                    for model in model_display_names.values()
                                ],
                                "2-S": [
                                    model
                                    for model in model_display_names.values()
                                ],
                                "Perturbed Examples": [
                                    "Gemini 2.5F",
                                    "Gemini 2.5F (T)",
                                ],
                            },
                        )
                    )
                    table = None
