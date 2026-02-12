"""Evaluate Ollama models on perturbation tasks."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from code.metrics.eval_metrics_latex import DeltaEvalTable
from code.metrics.metric_utils import (
    flip_label,
    convert_bool_label_to_str,
    check_file_exists_and_readable,
    get_exclusion_list,
)


def get_configuration() -> (
    Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]
):
    """Get configuration for perturbation evaluation.

    Returns:
        Tuple containing perturbations, models, and short perturbation names.
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

    models = [
        "deepseek-r1:32b",
        "deepseek-r1:32b_think",
        "qwen3:32b",
        "qwen3:32b_think",
        "llama3.3:70b",
        "mistral",
        "llama3.2:1b-instruct-q2_K",
    ]

    model_display_names = {
        "deepseek-r1:32b": "DeepSeek-R1-32B",
        "deepseek-r1:32b_think": "DeepSeek-32B (T)",
        "qwen3:32b": "Qwen3-32B",
        "qwen3:32b_think": "Qwen3-32B (T)",
        "llama3.3:70b": "Llama3.3-70B",
        "mistral": "Mistral-7B",
        "llama3.2:1b-instruct-q2_K": "Llama2-1B-Instruct",
    }

    short_pert_names = {
        "normalized_claim": "original",
        "approximation": "approx",
        "negative_number": "neg_num",
        "numeration": "num",
        "random_replacement": "rand_repl",
        "range": "range",
        "masking": "mask",
    }

    return perturbations, models, short_pert_names, model_display_names


def get_result_file_path(
    shot: str,
    perturbation: str,
    original_label: str,
    model: str,
    is_normalized: bool = False,
) -> str:
    """Get the path to a result file.

    Args:
        shot: The shot setting (e.g., "2-shot").
        perturbation: The perturbation type.
        original_label: The original label ("true" or "false").
        model: The model name.
        is_normalized: Whether to use numeration file for normalized claims.

    Returns:
        Path to the result file.
    """
    base_dir = f"results/ollama_runs/{shot}"

    if is_normalized:
        # For normalized claim, use numeration file
        return f"{base_dir}/numeration_{original_label}_{model}_results.jsonl"

    return f"{base_dir}/{perturbation}_{original_label}_{model}_results.jsonl"


def process_result_file(  # noqa: C901
    result_file: str,
    perturbation: str,
    original_label: str,
    perturbed_label: str,
) -> tuple[list[str], list[str], int, int, int]:
    """Process a single result file and extract labels.

    Args:
        result_file: Path to the result file to process.
        perturbation: The perturbation type being processed.
        original_label: The original label ("true" or "false").
        perturbed_label: The perturbed label ("true" or "false").

    Returns:
        Tuple containing true labels, predicted labels, line count,
        valid entries count, and invalid entries count.
    """
    true_labels: list[str] = []
    pred_labels: list[str] = []
    line_count = 0
    valid_entries = 0
    invalid_entries = 0
    exclusion_list = get_exclusion_list()
    # Check if file exists and is readable
    file_ok, error_msg = check_file_exists_and_readable(result_file)
    if not file_ok:
        print(f"  ❌ {error_msg}")
        return (
            true_labels,
            pred_labels,
            line_count,
            valid_entries,
            invalid_entries,
        )

    print("✅ File found and readable")

    try:
        with open(result_file, "r") as res_fp:
            for line in res_fp:
                line_count += 1
                try:
                    result_row = json.loads(line)
                    claim_id = int(result_row.get("claim_id", None))
                    if claim_id is None or claim_id in exclusion_list:
                        continue
                    pred_label_key = f"{perturbation}_{perturbed_label}"
                    if perturbation == "normalized_claim":
                        pred_label_key = "normalized_claim"

                    pred_label_bool = result_row.get(
                        f"{pred_label_key}_pred_label", None
                    )
                    if pred_label_bool is None:
                        print("pred is none for", result_row["claim_id"])
                        invalid_entries += 1
                        continue

                    pred_labels.append(
                        convert_bool_label_to_str(pred_label_bool)
                    )

                    if (
                        original_label == "true"
                        and perturbed_label == "false"
                        and perturbation != "normalized_claim"
                    ):
                        true_labels.append(flip_label(result_row["label"]))
                    else:
                        true_labels.append(result_row["label"])

                    valid_entries += 1

                except json.JSONDecodeError as e:
                    print(f"    Invalid JSON on line {line_count}: {e}")
                    continue

    except Exception as e:
        print(f"  ❌ Error processing file: {e}")

    return true_labels, pred_labels, line_count, valid_entries, invalid_entries


def evaluate_label_combination(
    original_label: str,
    perturbed_label: str,
    shot: str,
    perturbations: List[str],
    models: List[str],
    short_pert_names: dict[str, str],
    error_file,
    model_display_names: dict[str, str],
    table: DeltaEvalTable,
) -> None:
    """Evaluate a specific original-perturbed label combination.

    Args:
        original_label: The original label ("true" or "false").
        perturbed_label: The perturbed label ("true" or "false").
        shot: The shot setting (e.g., "2-shot").
        perturbations: Set of perturbation types.
        models: Set of model names.
        short_pert_names: Mapping of perturbation names to short names.
        error_file: File handle for error statistics.
        model_display_names: Mapping of model names to display names.
        table: DeltaEvalTable instance to store results.
    """
    for model in models:
        print(f"\nProcessing model: {model}")
        normalized_done = False

        for perturbation in perturbations:
            if perturbation == "normalized_claim" and normalized_done:
                continue

            print(f"  Checking perturbation: {perturbation}")

            # Determine if we're handling normalized claim
            is_normalized = perturbation == "normalized_claim"
            result_file = get_result_file_path(
                shot, perturbation, original_label, model, is_normalized
            )

            print(f"  Expected file: {result_file}")

            (
                true_labels,
                pred_labels,
                line_count,
                valid_entries,
                invalid_entries,
            ) = process_result_file(
                result_file, perturbation, original_label, perturbed_label
            )

            error_file.write(
                f"""{result_file}\t{line_count}\t{valid_entries}
                \t{invalid_entries}\n"""
            )

            if len(true_labels) == 0 or len(pred_labels) == 0:
                print("  ⚠️  No valid data found")
                continue

            if len(true_labels) != len(pred_labels):
                print(
                    f"""  ⚠️  Mismatch: {len(true_labels)} true labels vs
                     {len(pred_labels)} pred labels"""
                )
                continue

            if perturbation == "normalized_claim":
                normalized_done = True
                print(f"  ✅ Original done for {model}")

            table.add_result(
                model_display_names[model] + "_" + shot,
                short_pert_names[perturbation],
                true_labels,
                pred_labels,
            )

    # Generate and save LaTeX table


def main() -> None:
    """Evaluate Ollama models on various perturbation tasks.

    Loads results from JSONL files and computes accuracy metrics using
    DeltaEvalTable for different model and perturbation combinations.
    """
    (
        perturbations,
        models,
        short_pert_names,
        model_display_names,
    ) = get_configuration()

    original_labels = ["true", "false"]
    perturbed_labels = ["false", "true"]
    shots = ["0-shot", "2-shot", "neg-shot", "neg-same-type-shot"]

    latex_folder = "results/ollama_runs/eval_tables"

    with open("error_stats.txt", "w") as error_file:
        error_file.write(
            "result_file\tline_count\tvalid_entries\tinvalid_entries\n"
        )

        for original_label in original_labels:
            for perturbed_label in perturbed_labels:
                table = DeltaEvalTable(metrics=("accuracy",), digits=2)
                for shot in shots:
                    # Check if base directory exists
                    base_dir = Path("results/ollama_runs") / shot
                    if not base_dir.exists():
                        print(
                            f"Error: Base directory {base_dir} does not exist"
                        )
                        return
                    evaluate_label_combination(
                        original_label,
                        perturbed_label,
                        shot,
                        perturbations,
                        models,
                        short_pert_names,
                        error_file,
                        model_display_names,
                        table,
                    )
                    if table is None:
                        print("No table generated, skipping LaTeX output.")
                        continue
                qwen_models = [
                    model
                    for model in model_display_names.values()
                    if "qwen" in model.lower()
                ]
                perts = [
                    name
                    for name in short_pert_names.values()
                    if name != "original"
                ]
                latex = table.to_latex(
                    variations=perts,
                    section_labels={
                        shots[0]: list(model_display_names.values()),
                        shots[1]: list(model_display_names.values()),
                        shots[2]: qwen_models,
                        shots[3]: qwen_models,
                    },
                )
                latex = latex.replace(":", "-").replace("_", "-")
                print(latex)

                with open(
                    f"{latex_folder}/{original_label}-{perturbed_label}.tex",
                    "w",
                ) as f:
                    f.write(latex)


if __name__ == "__main__":
    main()
