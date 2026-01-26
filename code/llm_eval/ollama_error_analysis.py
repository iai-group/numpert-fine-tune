"""Extract and process token length data from Ollama model results.

Usage: python -m code.llm_eval.ollama_error_analysis
"""

import json
import os
from typing import Dict, Optional, Tuple

from code.metrics.eval_ollama_models import get_result_file_path


def get_required_data(
    line: str, perturb: str
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[bool],
    Optional[str],
]:
    """Extract required data fields from a JSON line.

    Args:
        line: JSON string containing claim data.
        perturb: Perturbation type identifier.

    Returns:
        Tuple containing claim_id, claim, evidence, reasoning_text,
        prompt_tokens_length, reasoning_tokens_length, ground_truth, prediction.
        Returns all None values if response is missing.
    """
    data = json.loads(line)
    claim_id = data["claim_id"]
    claim = data["claim"].strip().replace("\n", " ").replace("\r", "")
    evidence = data["doc"].strip().replace("\n", " ").replace("\r", "")

    key = f"{perturb}_false_ollama_response"
    response = data.get(key)

    if response is None:
        return None, None, None, None, None, None, None, None

    prompt_tokens_length = response.get("prompt_eval_count", "N/A")
    reasoning_tokens_length = response.get("eval_count", "N/A")
    ground_truth = data["label"]

    # Flip label since T2F
    ground_truth = False if ground_truth == "True" else True
    prediction = data.get(f"{perturb}_false_pred_label", "N/A")
    reasoning_text = (
        response.get("message", {})
        .get("thinking", "")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    return (
        claim_id,
        claim,
        evidence,
        reasoning_text,
        prompt_tokens_length,
        reasoning_tokens_length,
        ground_truth,
        prediction,
    )


def process_file(file_path: str, perturb: str) -> Dict[str, Dict]:
    """Process a result file and extract token length data.

    Args:
        file_path: Path to the JSONL file to process.
        perturb: Perturbation type identifier.

    Returns:
        Dictionary mapping claim_id to extracted data.
    """
    results = {}
    output = file_path.replace(".jsonl", "_token_length.tsv")

    with open(output, "w") as out_f:
        out_f.write(
            "claim_id\tclaim\tdoc\tthoughts\tprompt_tokens_length\t"
            "reasoning_tokens_length\tground_truth\tprediction\n"
        )

        with open(file_path, "r") as f:
            for line in f:
                extracted_data = get_required_data(line, perturb)
                (
                    claim_id,
                    claim,
                    evidence,
                    reasoning_text,
                    prompt_tokens_length,
                    reasoning_tokens_length,
                    ground_truth,
                    prediction,
                ) = extracted_data

                if claim_id is None:
                    continue

                results[claim_id] = {
                    "claim": claim,
                    "evidence": evidence,
                    "thoughts": reasoning_text,
                    "prompt_tokens_length": prompt_tokens_length,
                    "reasoning_tokens_length": reasoning_tokens_length,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }

                out_f.write(
                    f"{claim_id}\t{claim}\t{evidence}\t{reasoning_text}\t"
                    f"{prompt_tokens_length}\t{reasoning_tokens_length}\t"
                    f"{ground_truth}\t{prediction}\n"
                )

    return results


def main(
    perturb: str, label: str, model_name: str, analyze_errors: bool
) -> None:
    """Generate combined token length analysis for matching predictions.

    Args:
        perturb: Perturbation type identifier.
        label: Label type (e.g., "true").
        model_name: Name of the model being analyzed.
        analyze_errors: If True, analyze errors; else analyze correct preds.
    """
    file_path1 = get_result_file_path(
        shot="0-shot",
        perturbation=perturb,
        original_label=label,
        model=model_name,
        is_normalized=False,
    )
    file_path2 = get_result_file_path(
        shot="neg-shot",
        perturbation=perturb,
        original_label=label,
        model=model_name,
        is_normalized=False,
    )

    results1 = process_file(file_path1, perturb)
    results2 = process_file(file_path2, perturb)

    if analyze_errors:
        combined_file = file_path1.replace(".jsonl", "_combined_errors.tsv")
    else:
        combined_file = file_path1.replace(".jsonl", "_combined_correct.tsv")
    summary_file = "results/ollama_runs/token_length_summary.tsv"
    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a") as summary_f:
        if not file_exists:
            summary_f.write(
                "perturbation\tanalyze_errors\tavg_prompt_0shot\tavg_reasoning_0shot\t"
                "avg_prompt_negshot\tavg_reasoning_negshot\n"
            )
        with open(combined_file, "w") as out_f:
            header = (
                "claim_id\tclaim\tdoc\tthoughts_0shot\tprompt_tokens_length_0shot\t"
                "reasoning_tokens_length_0shot\tground_truth_0shot\tprediction_0shot\t"  # noqa: E501
                "thoughts_negshot\tprompt_tokens_length_negshot\t"
                "reasoning_tokens_length_negshot\tground_truth_negshot\t"
                "prediction_negshot\n"
            )  # noqa: E501
            out_f.write(header)
            prompt_token_length_0shot = 0
            reasoning_token_length_0shot = 0
            prompt_token_length_negshot = 0
            reasoning_token_length_negshot = 0
            num_entries = 0
            for claim_id, data1 in results1.items():
                data2 = results2.get(claim_id)
                if (
                    analyze_errors
                    and data1["ground_truth"] != data1["prediction"]
                    and data2["ground_truth"] == data2["prediction"]
                    or not analyze_errors
                    and data1["ground_truth"] == data1["prediction"]
                    and data2["ground_truth"] is not None
                ):
                    prompt_token_length_0shot += data1["prompt_tokens_length"]
                    reasoning_token_length_0shot += data1[
                        "reasoning_tokens_length"
                    ]
                    prompt_token_length_negshot += data2["prompt_tokens_length"]
                    reasoning_token_length_negshot += data2[
                        "reasoning_tokens_length"
                    ]
                    num_entries += 1
                    out_f.write(
                        f"{claim_id}\t{data1['claim']}\t{data1['evidence']}\t"
                        f"{data1['thoughts']}\t{data1['prompt_tokens_length']}\t"
                        f"{data1['reasoning_tokens_length']}\t{data1['ground_truth']}\t"  # noqa: E501
                        f"{data1['prediction']}\t{data2['thoughts']}\t"
                        f"{data2['prompt_tokens_length']}\t{data2['reasoning_tokens_length']}\t"  # noqa: E501
                        f"{data2['ground_truth']}\t{data2['prediction']}\n"
                    )
            # Calculate averages
            avg_prompt_0shot = (
                prompt_token_length_0shot / num_entries
                if num_entries > 0
                else 0
            )
            avg_reasoning_0shot = (
                reasoning_token_length_0shot / num_entries
                if num_entries > 0
                else 0
            )
            avg_prompt_negshot = (
                prompt_token_length_negshot / num_entries
                if num_entries > 0
                else 0
            )
            avg_reasoning_negshot = (
                reasoning_token_length_negshot / num_entries
                if num_entries > 0
                else 0
            )

            # Write to summary TSV file

            summary_f.write(
                f"{perturb}\t{analyze_errors}\t{avg_prompt_0shot}\t{avg_reasoning_0shot}\t"
                f"{avg_prompt_negshot}\t{avg_reasoning_negshot}\n"
            )


if __name__ == "__main__":
    perturbations = [
        "approximation",
        "negative_number",
        "numeration",
        "random_replacement",
        "range",
        "masking",
    ]
    for analyze_errors in [True, False]:
        for perturb in perturbations:
            main(perturb, "true", "qwen3:32b_think", analyze_errors)
