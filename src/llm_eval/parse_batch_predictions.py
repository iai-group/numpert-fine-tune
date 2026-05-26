#!/usr/bin/env python3
"""Parse Gemini (OpenAI-batch shaped) prediction outputs and score them.

Reads every ``*.jsonl`` file in an input directory -- by default
``predictions/gemini-flash/test_split`` -- and produces the same
metrics report, parse-failure dump and predictions CSV as
``src/llm_finetune/utils/inference/metrics.py`` so the Gemini results
are directly comparable with the quantized / fine-tuned model runs.

Each input record is expected to follow the batch-response shape
produced by ``src/llm_eval/openAI.py`` (also used for Gemini)::

    {
        "custom_id": "...",
        "response": {"status_code": 200, "body": {"choices": [{
            "message": {"content": "{\\"label\\": true}"}, ...
        }]}},
        "error": null,
        "meta": {
            "claim_id": "test_000003",
            "label": "False",
            "variant_key": "perturb_type1_False",
            "perturbation_type": "perturb_type1",
            "dataset_type": "approximation_False",
            "claim": "..."
        }
    }
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from src.llm_finetune.utils.inference.metrics import (  # noqa: E402
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
    save_parse_failures,
    save_predictions_csv,
)

LABEL_TRUE = '{"label": true}'
LABEL_FALSE = '{"label": false}'


def normalize_ground_truth(label: Any) -> str:
    """Convert a raw ``"True"``/``"False"`` meta label to the JSON-string
    form used throughout the metrics module."""
    s = str(label).strip().lower()
    if s == "true":
        return LABEL_TRUE
    if s == "false":
        return LABEL_FALSE
    raise ValueError(f"Unrecognized ground-truth label: {label!r}")


def derive_ground_truth(original_label: Any, perturbation_type: str) -> str:
    """Return the post-perturbation ground-truth label.

    ``meta.label`` in the Gemini batch output carries the *original*
    claim's label (the source file's label). The actual evaluated
    claim's label depends on the perturbation:

    - ``original_normalized``: unchanged (baseline)
    - ``perturb_type1``: unchanged (structure-preserving)
    - ``perturb_type2``: always ``False`` (value-modifying)
    - ``perturb_type3_*``: always ``False`` (obfuscation)
    """
    original = normalize_ground_truth(original_label)
    if perturbation_type in ("original_normalized", "perturb_type1"):
        return original
    if perturbation_type == "perturb_type2" or perturbation_type.startswith(
        "perturb_type3_"
    ):
        return LABEL_FALSE
    return original


def extract_predicted_label(content: Optional[str]) -> Optional[str]:
    """Best-effort extraction of ``{"label": true|false}`` from raw output.

    Returns one of :data:`LABEL_TRUE` / :data:`LABEL_FALSE`, or ``None``
    if the response cannot be parsed as a label.
    """
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fall back to substring matching for slightly malformed outputs.
        lowered = content.lower()
        has_true = '"label": true' in lowered or '"label":true' in lowered
        has_false = '"label": false' in lowered or '"label":false' in lowered
        if has_true and not has_false:
            return LABEL_TRUE
        if has_false and not has_true:
            return LABEL_FALSE
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("label")
    if isinstance(value, bool):
        return LABEL_TRUE if value else LABEL_FALSE
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "true":
            return LABEL_TRUE
        if s == "false":
            return LABEL_FALSE
    return None


def _content_from_response(record: Dict[str, Any]) -> Optional[str]:
    """Return the assistant message content from a batch-response record."""
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    if response.get("status_code") != 200:
        return None
    body = response.get("body")
    if not isinstance(body, dict):
        return None
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else None


def parse_record(
    record: Dict[str, Any], index: int
) -> Optional[Tuple[Dict, Dict]]:
    """Parse one JSONL record into ``(result, raw_output)`` for metrics.

    The ``raw_output`` is shaped to match the inference_details record
    written by the qwen / unsloth runs (``index``, ``claim_id``,
    ``perturbation_type``, ``label``, ``raw_response``,
    ``parsed_successfully``, ``predicted_label``, ``source_file``) so
    Gemini outputs are directly comparable. The original ``custom_id``
    and ``claim`` text are retained as extra fields for traceability.

    Returns ``None`` for records missing the required meta fields.
    """
    meta = record.get("meta") or {}
    claim_id = meta.get("claim_id")
    label = meta.get("label")
    perturbation_type = meta.get("perturbation_type")
    dataset_type = meta.get("dataset_type")
    if claim_id is None or label is None or perturbation_type is None:
        return None

    source_file = f"{dataset_type}.json" if dataset_type else ""
    content = _content_from_response(record)
    predicted_label = extract_predicted_label(content)
    gt_label = derive_ground_truth(label, perturbation_type)

    result: Dict[str, Any] = {
        "claim_id": claim_id,
        "label": gt_label,
        "prediction": predicted_label,
        "perturbation_type": perturbation_type,
    }
    if source_file:
        result["source_file"] = source_file

    raw_output: Dict[str, Any] = {
        "index": index,
        "claim_id": claim_id,
        "perturbation_type": perturbation_type,
        "label": gt_label,
        "raw_response": content if content is not None else "",
        "parsed_successfully": predicted_label is not None,
        "predicted_label": predicted_label,
    }
    if source_file:
        raw_output["source_file"] = source_file
    raw_output["custom_id"] = record.get("custom_id")
    raw_output["claim"] = meta.get("claim", "")
    return result, raw_output


def load_reference_triples(path: str) -> set:
    """Load ``(claim_id, source_file, perturbation_type)`` row IDs from a
    qwen-style ``inference_details.jsonl`` reference run.

    Records missing any of those fields are skipped silently.
    """
    triples: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = r.get("claim_id")
            sf = r.get("source_file")
            pt = r.get("perturbation_type")
            if cid is None or sf is None or pt is None:
                continue
            triples.add((cid, sf, pt))
    return triples


def load_jsonl_dir(
    input_dir: str,
    excluded_perturbation_types: Optional[Sequence[str]] = None,
    reference_triples: Optional[set] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Read every ``*.jsonl`` in *input_dir* (non-recursive).

    Deduplicates on ``custom_id`` (which the upstream runner builds as
    ``claim_id-label-variant_key-dataset_type``). A custom_id maps
    1:1 to the qwen inference_details row identity
    ``(claim_id, source_file, perturbation_type)``, so this dedup
    drops only the within-file repeats produced by the obfuscation
    expansion bug (same baseline emitted once per obfuscation subtype)
    while preserving legitimate per-file evaluations.

    *excluded_perturbation_types* drops any record whose
    ``perturbation_type`` is in the iterable (e.g. ``("perturb_type3_removed",)``
    to align with a qwen test split that omits that subtype).

    *reference_triples* — if provided — restricts the output to
    records whose ``(claim_id, source_file, perturbation_type)`` is in
    that set. Use it to exactly match the row coverage of a reference
    qwen ``inference_details.jsonl`` (different model configs use
    differently filtered test splits).
    """
    paths = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No '*.jsonl' files found in '{input_dir}'.")

    excluded = set(excluded_perturbation_types or ())

    results: List[Dict] = []
    raw_outputs: List[Dict] = []
    skipped = 0
    seen_custom_ids: set = set()
    dup_drops = 0
    excluded_drops = 0
    not_in_reference = 0
    matched_triples: set = set()
    for path in paths:
        file_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"  skip {os.path.basename(path)}:{line_no} "
                        f"(invalid JSON: {e})",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue
                custom_id = record.get("custom_id")
                if custom_id is not None and custom_id in seen_custom_ids:
                    dup_drops += 1
                    continue
                parsed = parse_record(record, index=len(results))
                if parsed is None:
                    skipped += 1
                    continue
                result, raw_output = parsed
                if result["perturbation_type"] in excluded:
                    excluded_drops += 1
                    continue
                if reference_triples is not None:
                    triple = (
                        result["claim_id"],
                        result.get("source_file", ""),
                        result["perturbation_type"],
                    )
                    if triple not in reference_triples:
                        not_in_reference += 1
                        continue
                    matched_triples.add(triple)
                if custom_id is not None:
                    seen_custom_ids.add(custom_id)
                results.append(result)
                raw_outputs.append(raw_output)
                file_count += 1
        print(f"  loaded {file_count:>5d} records from {path}")
    if skipped:
        print(f"Skipped {skipped} records (missing meta or invalid JSON).")
    if dup_drops:
        print(
            f"Deduped {dup_drops} duplicate custom_id records "
            f"(within-file repeats — kept first)."
        )
    if excluded_drops:
        print(
            f"Excluded {excluded_drops} records with "
            f"perturbation_type in {sorted(excluded)}."
        )
    if reference_triples is not None:
        missing = len(reference_triples) - len(matched_triples)
        print(
            f"Reference filter: kept {len(matched_triples)}/"
            f"{len(reference_triples)} reference rows; dropped "
            f"{not_in_reference} gemini records not in reference."
        )
        if missing:
            print(
                f"  Warning: {missing} reference rows had no matching "
                f"gemini record (gemini missing those evaluations)."
            )
    return results, raw_outputs


def save_inference_details(
    raw_outputs: List[Dict],
    output_dir: str,
    filename: str = "inference_details.jsonl",
) -> str:
    """Save per-record inference details as JSONL (mirrors unsloth run)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for entry in raw_outputs:
            f.write(json.dumps(entry) + "\n")
    return path


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse Gemini batch prediction JSONL files in --input_dir "
            "and write a metrics report compatible with "
            "src/llm_finetune/utils/inference/metrics.py."
        )
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default="predictions/gemini-flash/test_split",
        help=(
            "Directory containing JSONL prediction files "
            "(default: predictions/gemini-flash/test_split)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="results/inference/gemini-flash/test_split",
        help=(
            "Directory in which to write metrics.json, predictions.csv, "
            "inference_details.jsonl and parse_failures.json."
        ),
    )
    parser.add_argument(
        "--show_classification_reports",
        action="store_true",
        help=(
            "Include the full per-class sklearn classification reports "
            "in the printed summary."
        ),
    )
    parser.add_argument(
        "--exclude_perturbation_type",
        action="append",
        default=None,
        help=(
            "Drop records whose 'perturbation_type' matches this value. "
            "Pass multiple times to exclude several types. Ignored when "
            "--reference_inference_details is supplied (the reference "
            "already constrains perturbation types)."
        ),
    )
    parser.add_argument(
        "--reference_inference_details",
        default=(
            "results/inference/unquant/Qwen3-0.6b/"
            "Qwen3-0.6B-all_perturb_no_f_type2/inference_details.jsonl"
        ),
        help=(
            "Path to a qwen inference_details.jsonl whose "
            "(claim_id, source_file, perturbation_type) rows define "
            "the canonical test-set coverage. Gemini records outside "
            "those rows are dropped so per-group totals match the "
            "reference qwen run exactly. Pass an empty string to "
            "disable filtering."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_arguments()

    reference_triples: Optional[set] = None
    if args.reference_inference_details:
        print(
            f"Loading reference rows from "
            f"'{args.reference_inference_details}'"
        )
        reference_triples = load_reference_triples(
            args.reference_inference_details
        )
        print(f"  reference defines {len(reference_triples)} unique rows")

    print(f"Reading predictions from '{args.input_dir}'")
    results, raw_outputs = load_jsonl_dir(
        args.input_dir,
        excluded_perturbation_types=args.exclude_perturbation_type,
        reference_triples=reference_triples,
    )
    if not results:
        print(f"No records loaded from '{args.input_dir}'. Nothing to do.")
        return
    print(f"Loaded {len(results)} records total.\n")

    report = compute_metrics(results)
    print_metrics_report(
        report,
        show_classification_reports=args.show_classification_reports,
    )

    metrics_path = save_metrics_report(report, args.output_dir)
    details_path = save_inference_details(raw_outputs, args.output_dir)
    csv_path = save_predictions_csv(results, raw_outputs, args.output_dir)
    failures_path = save_parse_failures(results, raw_outputs, args.output_dir)

    print(f"Wrote metrics to            {metrics_path}")
    print(f"Wrote inference details to  {details_path}")
    print(f"Wrote predictions CSV to    {csv_path}")
    if failures_path:
        print(f"Wrote parse failures to     {failures_path}")
    else:
        print("No parse failures (no parse_failures.json written).")


if __name__ == "__main__":
    main()
