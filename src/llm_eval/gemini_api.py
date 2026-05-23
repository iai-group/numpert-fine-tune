#!/usr/bin/env python3
"""Process perturbed-data test inputs with Google Gemini.

Auth defaults to ADC + Vertex AI. The google-auth library picks up
``~/.config/gcloud/application_default_credentials.json`` automatically
(created by ``gcloud auth application-default login``). Pass ``--project``
or set ``GOOGLE_CLOUD_PROJECT``; ``--location`` defaults to ``us-central1``.
Pass ``--api_key`` (or set ``GOOGLE_API_KEY``) to switch to the Gemini
Developer API instead.

Supports two input modes (auto-detected by extension):

  * ``*.jsonl`` — OpenAI batch format (one request per line) as produced by
    ``json_to_jsonl.py``. The ``body.messages`` field is consumed as-is.
  * ``*.json``  — A raw test_split JSON file (a list of entries with
    ``claim_id``, ``label``, ``doc``, ``normalized_claim`` and any number of
    ``perturb_type*`` keys). For each entry we emit one request per variant
    (``normalized_claim`` + every ``perturb_type*`` field), building the prompt
    via :mod:`src.numpert_prompt.prompt` so the user only needs to pick a
    ``--prompt_type`` (default: ``zero_shot``).
"""

import argparse
import json
import sys
import time
import os
from typing import Dict, List, Optional, Any, Set, Tuple

import google.genai as genai  # type: ignore[import]
from google.genai import types  # type: ignore[import]

# Make ``src.*`` importable regardless of where the script is invoked from.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from src.numpert_prompt.prompt import (  # noqa: E402
    PromptType,
    get_fact_check_system_prompt,
    get_fact_check_user_prompt,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (  # noqa: E402, E501
    PerturbationType,
)


# Suffixes that the test_split JSON keys append to the canonical
# perturbation-type name. ``perturb_type1_True`` / ``perturb_type1_False`` both
# correspond to the canonical ``perturb_type1``.
_LABEL_SUFFIXES = ("_True", "_False")


def variant_key_to_perturbation_type(key: str) -> str:
    """Translate a raw JSON variant key into a canonical PerturbationType value.

    Examples::

        normalized_claim            -> original_normalized
        original_normalized         -> original_normalized
        perturb_type1_True          -> perturb_type1
        perturb_type1_False         -> perturb_type1
        perturb_type3_random_False  -> perturb_type3_random
        perturb_type3_masked_True   -> perturb_type3_masked
        perturb_type3_removed_False -> perturb_type3_removed

    Unknown keys are returned unchanged so the caller can decide how to handle
    them (e.g. surface a warning).
    """
    if key in ("normalized_claim", "original_normalized"):
        return PerturbationType.ORIGINAL_NORMALIZED.value

    canonical = key
    for suffix in _LABEL_SUFFIXES:
        if canonical.endswith(suffix):
            canonical = canonical[: -len(suffix)]
            break

    try:
        return PerturbationType(canonical).value
    except ValueError:
        return canonical


def iter_variant_keys(entry: Dict[str, Any]):
    """Yield (variant_key, claim_text) pairs for one test_split entry.

    Order: baseline first (``original_normalized`` if present, else
    ``normalized_claim``), then every ``perturb_type*`` key in the order they
    appear on the entry.
    """
    baseline = entry.get("original_normalized") or entry.get("normalized_claim")
    if baseline:
        baseline_key = (
            "original_normalized"
            if "original_normalized" in entry
            else "normalized_claim"
        )
        yield baseline_key, baseline

    for key, value in entry.items():
        if key.startswith("perturb_type") and isinstance(value, str):
            yield key, value


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate claims with Google Gemini. Accepts OpenAI-batch JSONL "
            "or raw test_split JSON (with normalized_claim + perturb_type* "
            "fields). For JSON inputs, the prompt is built from "
            "src/numpert_prompt/prompt.py."
        )
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help=(
            "Path to a JSONL/JSON input file, OR a directory containing such "
            "files (every .json/.jsonl in the dir will be processed)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help=(
            "Output path. Must be a file when -i is a file, or a directory "
            "(created if missing) when -i is a directory; in directory mode "
            "each input file gets its own '<stem>.jsonl' output."
        ),
    )
    parser.add_argument(
        "--api_key",
        help=(
            "Google API key. If set (or GOOGLE_API_KEY is in the env), the "
            "client uses the Gemini Developer API. Otherwise the default is "
            "ADC + Vertex AI (see --project / --location)."
        ),
    )
    parser.add_argument(
        "--project",
        help=(
            "GCP project for Vertex AI. Defaults to the GOOGLE_CLOUD_PROJECT "
            "env var. Required when running with ADC (the default)."
        ),
    )
    parser.add_argument(
        "--location",
        default=None,
        help=(
            "Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION env var or "
            "'us-central1'."
        ),
    )
    parser.add_argument(
        "--num_elements",
        type=int,
        help="Max requests to evaluate (counted after JSON expansion).",
    )
    parser.add_argument(
        "--model_name", default="gemini-2.5-flash", help="Gemini model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation (0.0 to 2.0).",
    )
    parser.add_argument(
        "--prompt_type",
        default=PromptType.ZERO_SHOT.value,
        choices=[pt.value for pt in PromptType],
        help=(
            "Prompt template to use when --input_file is a raw test_split "
            "JSON. Ignored for JSONL inputs (their messages are taken as-is)."
        ),
    )
    parser.add_argument(
        "--show_model_output",
        action="store_true",
        help="Print the model's input messages",
    )
    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Enable reasoning mode with thinking",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=8192,
        help="Thinking budget (0=no thinking, 1=dynamic)",
    )
    parser.add_argument(
        "--show_thoughts",
        action="store_true",
        help="Include model thoughts in output and display",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records whose custom_id is already in the output file.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save progress every N records",
    )
    return parser.parse_args()


def read_jsonl(
    file_path: str, num_elements: Optional[int] = None
) -> List[Dict]:
    """Read an OpenAI-batch JSONL file."""
    data: List[Dict] = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if num_elements and i >= num_elements:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                continue
    return data


def read_test_split_json(
    file_path: str,
    model_name: str,
    prompt_type: PromptType,
    num_elements: Optional[int] = None,
) -> List[Dict]:
    """Read a raw test_split JSON and expand each entry into per-variant rows.

    Mirrors ``create_jsonl_from_test_split`` in ``json_to_jsonl.py``: each
    entry yields one request for ``normalized_claim`` plus one per
    ``perturb_type*`` key. The variant key is encoded in the custom_id so the
    source claim is recoverable downstream.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(
            f"Expected '{file_path}' to contain a JSON list of entries."
        )

    dataset_type = os.path.splitext(os.path.basename(file_path))[0]
    system_prompt = get_fact_check_system_prompt()

    records: List[Dict] = []
    skipped = 0
    for i, entry in enumerate(entries):
        claim_id = entry.get("claim_id")
        label = entry.get("label")
        evidence = entry.get("doc")
        baseline_claim = entry.get("original_normalized") or entry.get(
            "normalized_claim"
        )

        if not all([claim_id is not None, evidence, baseline_claim]):
            print(
                f"Warning: skipping entry {i} in '{file_path}' (missing "
                "claim_id/doc/original_normalized|normalized_claim)."
            )
            skipped += 1
            continue

        for variant_key, claim in iter_variant_keys(entry):
            if not claim:
                skipped += 1
                continue
            perturbation_type = variant_key_to_perturbation_type(variant_key)
            user_content = get_fact_check_user_prompt(
                claim=claim, evidence=evidence, prompt_type=prompt_type
            )
            records.append(
                {
                    "custom_id": (
                        f"{claim_id}-{label}-{variant_key}-{dataset_type}"
                    ),
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                    },
                    "_meta": {
                        "claim_id": claim_id,
                        "label": label,
                        "variant_key": variant_key,
                        "perturbation_type": perturbation_type,
                        "dataset_type": dataset_type,
                        "claim": claim,
                    },
                }
            )
            if num_elements and len(records) >= num_elements:
                return records

    if skipped:
        print(f"Skipped {skipped} entries/variants in '{file_path}'.")
    return records


def load_input(input_file: str, args: argparse.Namespace) -> List[Dict]:
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(input_file, args.num_elements)
    if ext == ".json":
        return read_test_split_json(
            input_file,
            model_name=args.model_name,
            prompt_type=PromptType(args.prompt_type),
            num_elements=args.num_elements,
        )
    raise ValueError(
        f"Unsupported input extension '{ext}'. Use .json or .jsonl."
    )


_SUPPORTED_INPUT_EXTS = (".json", ".jsonl")


def resolve_io_pairs(
    input_path: str, output_path: str
) -> List[Tuple[str, str]]:
    """Resolve ``-i`` and ``-o`` into a list of (input_file, output_file).

    Rules:
      * file → file: returned as-is.
      * dir  → dir: every ``.json``/``.jsonl`` under the input dir is paired
        with an output ``.jsonl`` of the same stem in the output dir.
      * dir  → file: rejected (ambiguous).
      * file → dir : rejected (ambiguous).
    """
    if os.path.isdir(input_path):
        json_files = sorted(
            f
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_INPUT_EXTS
        )
        if not json_files:
            raise ValueError(
                f"No .json/.jsonl files found in directory '{input_path}'."
            )
        if os.path.exists(output_path) and not os.path.isdir(output_path):
            raise ValueError(
                f"Input '{input_path}' is a directory but output "
                f"'{output_path}' is an existing file. Output must be a "
                "directory in directory mode."
            )
        os.makedirs(output_path, exist_ok=True)
        pairs: List[Tuple[str, str]] = []
        for fname in json_files:
            stem = os.path.splitext(fname)[0]
            pairs.append(
                (
                    os.path.join(input_path, fname),
                    os.path.join(output_path, f"{stem}.jsonl"),
                )
            )
        return pairs

    if os.path.isdir(output_path):
        raise ValueError(
            f"Input '{input_path}' is a file but output '{output_path}' is a "
            "directory. Specify an output file path."
        )
    return [(input_path, output_path)]


def get_processed_ids(output_file: str) -> Set[str]:
    """Return the set of custom_ids already present in the output file."""
    processed_ids: Set[str] = set()
    if not os.path.exists(output_file):
        return processed_ids

    with open(output_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if "custom_id" in record:
                processed_ids.add(record["custom_id"])
    return processed_ids


def save_checkpoint(
    output_file: str, results: List[Dict], append_mode: bool = False
) -> None:
    """Write a batch of results to the output file.

    Creates the parent directory if it doesn't exist so callers don't need to
    pre-create the tree.
    """
    parent = os.path.dirname(output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "a" if append_mode else "w"
    with open(output_file, mode) as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def extract_system_and_user(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], str]:
    """Split OpenAI messages into a system_instruction and user content.

    Gemini natively supports a system_instruction config, so we pass the
    system prompt there instead of flattening it into the user turn with a
    ``System:`` prefix.
    """
    system_parts: List[str] = []
    user_parts: List[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"Assistant: {content}")

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    user_prompt = "\n\n".join(user_parts)
    return system_instruction, user_prompt


def process_with_gemini(  # noqa: C901
    client: genai.Client,
    system_instruction: Optional[str],
    user_prompt: str,
    model_name: str,
    temperature: float,
    use_reasoning: bool,
    thinking_budget: int,
    show_thoughts: bool,
) -> Dict[str, Any]:
    """Send a single request to Gemini and return a structured result."""
    thoughts = ""
    answer = ""
    thoughts_tokens = 0
    output_tokens = 0
    total_tokens = 0
    last_chunk = None

    try:
        config_kwargs: Dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if use_reasoning:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            )
        config = types.GenerateContentConfig(**config_kwargs)

        for chunk in client.models.generate_content_stream(
            model=model_name, contents=user_prompt, config=config
        ):
            last_chunk = chunk

            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    elif hasattr(part, "thought") and part.thought:
                        if show_thoughts and not thoughts:
                            print("\n[Thoughts:]", end=" ")
                        if show_thoughts:
                            print(part.text, end="")
                        thoughts += part.text
                    else:
                        if not answer and show_thoughts:
                            print("\n[Answer:]", end=" ")
                        if show_thoughts:
                            print(part.text, end="")
                        answer += part.text

        if (
            last_chunk
            and hasattr(last_chunk, "usage_metadata")
            and last_chunk.usage_metadata
        ):
            if hasattr(last_chunk.usage_metadata, "thoughts_token_count"):
                thoughts_tokens = (
                    last_chunk.usage_metadata.thoughts_token_count or 0
                )
            if hasattr(last_chunk.usage_metadata, "total_token_count"):
                total_tokens = last_chunk.usage_metadata.total_token_count or 0
            elif hasattr(last_chunk.usage_metadata, "candidates_token_count"):
                total_tokens = (
                    last_chunk.usage_metadata.candidates_token_count or 0
                )
            output_tokens = max(0, total_tokens - thoughts_tokens)

            if show_thoughts:
                print(
                    f"\n[Debug - Raw token counts: total={total_tokens}, "
                    f"thoughts={thoughts_tokens}, "
                    f"calculated_output={output_tokens}]"
                )

        return {
            "answer": answer.strip(),
            "thoughts": thoughts,
            "thoughts_tokens": thoughts_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error": None,
        }

    except Exception as e:
        return {
            "answer": "",
            "thoughts": "",
            "thoughts_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
        }


def create_output_record(
    custom_id: str,
    result: Dict[str, Any],
    model_name: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSONL output record (OpenAI batch shape + optional meta)."""
    if result["error"]:
        record = {
            "custom_id": custom_id,
            "response": None,
            "error": {
                "message": result["error"],
                "type": "api_error",
                "code": None,
            },
        }
        if meta:
            record["meta"] = meta
        return record

    usage: Dict[str, Any] = {
        "completion_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"],
    }
    if result["thoughts_tokens"] > 0:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": result["thoughts_tokens"],
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }

    response_body: Dict[str, Any] = {
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["answer"],
                    "refusal": None,
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "service_tier": "default",
        "system_fingerprint": None,
    }

    if result["thoughts"] and result["thoughts"].strip():
        response_body["choices"][0]["message"]["thoughts"] = result["thoughts"]

    record = {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": response_body},
        "error": None,
    }
    if meta:
        record["meta"] = meta
    return record


_DEFAULT_ADC_PATH = os.path.expanduser(
    "~/.config/gcloud/application_default_credentials.json"
)


def make_client(args: argparse.Namespace) -> genai.Client:
    """Build a Gemini client.

    Default: ADC + Vertex AI (uses
    ``~/.config/gcloud/application_default_credentials.json`` automatically
    via google-auth). Set ``--api_key`` or ``GOOGLE_API_KEY`` to use the
    Gemini Developer API instead.
    """
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("Auth: Gemini Developer API (api_key)")
        return genai.Client(api_key=api_key)

    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = (
        args.location
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    )
    if not project:
        print(
            "Error: ADC mode needs a GCP project. Pass --project or set "
            "GOOGLE_CLOUD_PROJECT. Alternatively pass --api_key to use the "
            "Gemini Developer API."
        )
        sys.exit(1)

    adc_path = (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or _DEFAULT_ADC_PATH
    )
    print(
        "Auth: ADC + Vertex AI "
        f"(project={project}, location={location})\n"
        f"      Credentials: {adc_path}"
    )
    if not os.path.exists(adc_path):
        print(
            f"Warning: ADC credentials not found at '{adc_path}'. "
            "Run `gcloud auth application-default login` to create them, "
            "or set GOOGLE_APPLICATION_CREDENTIALS to an alternate path."
        )
    return genai.Client(vertexai=True, project=project, location=location)


def run_inference_on_file(  # noqa: C901
    client: genai.Client,
    input_file: str,
    output_file: str,
    args: argparse.Namespace,
) -> None:
    """Run inference for a single (input_file → output_file) pair."""
    print("\n" + "=" * 80)
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print("=" * 80)

    input_data = load_input(input_file, args)
    print(f"Loaded {len(input_data)} requests")
    if input_file.endswith(".json"):
        print(f"Prompt type: {args.prompt_type}")

    processed_ids: Set[str] = set()
    append_mode = False
    if args.resume:
        processed_ids = get_processed_ids(output_file)
        if processed_ids:
            print(
                f"Resume: found {len(processed_ids)} already-processed records"
            )
            append_mode = True
        else:
            print("Resume: no existing output found, starting fresh.")

    remaining_records: List[Dict] = []
    skipped_count = 0
    for idx, record in enumerate(input_data):
        custom_id = record.get("custom_id", f"record_{idx}")
        if custom_id in processed_ids:
            skipped_count += 1
        else:
            remaining_records.append(record)

    if skipped_count:
        print(f"Skipping {skipped_count} already-processed records")

    if not remaining_records:
        print("All records have already been processed for this file.")
        return

    print(f"Processing {len(remaining_records)} remaining records")
    print(f"Using temperature: {args.temperature}")

    results_batch: List[Dict] = []
    processed_count = 0

    try:
        for i, record in enumerate(remaining_records):
            custom_id = record.get(
                "custom_id",
                f"record_{len(input_data) - len(remaining_records) + i}",
            )
            current_position = len(input_data) - len(remaining_records) + i + 1
            print(
                f"\nProcessing {current_position}/{len(input_data)}: "
                f"{custom_id}"
            )

            body = record.get("body", {})
            messages = body.get("messages", [])
            meta = record.get("_meta")

            if args.show_model_output:
                print("Messages:", json.dumps(messages, indent=2))

            system_instruction, user_prompt = extract_system_and_user(messages)

            if args.show_model_output:
                print("System instruction:", system_instruction)
                print("User prompt:", user_prompt)

            result = process_with_gemini(
                client=client,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                model_name=args.model_name,
                temperature=args.temperature,
                use_reasoning=args.use_reasoning,
                thinking_budget=args.thinking_budget,
                show_thoughts=args.show_thoughts,
            )

            output_record = create_output_record(
                custom_id, result, args.model_name, meta=meta
            )
            results_batch.append(output_record)
            processed_count += 1

            if args.show_thoughts and result["thoughts_tokens"] > 0:
                print(
                    f"\n[Tokens - Thoughts: {result['thoughts_tokens']}, "
                    f"Output: {result['output_tokens']}]"
                )

            if processed_count % args.checkpoint_interval == 0:
                print(
                    f"\n[Checkpoint] Saving progress after "
                    f"{processed_count} records..."
                )
                save_checkpoint(output_file, results_batch, append_mode)
                results_batch = []
                append_mode = True

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(
            f"\n\nInterrupted! Saving progress after "
            f"{processed_count} records..."
        )
        if results_batch:
            save_checkpoint(output_file, results_batch, append_mode)
        print("Progress saved. You can resume later using --resume flag.")
        sys.exit(1)

    except Exception:
        print(f"\nError on '{input_file}'. Saving partial results...")
        if results_batch:
            save_checkpoint(output_file, results_batch, append_mode)
        print("Progress saved. You can resume later using --resume flag.")
        raise

    if results_batch:
        print(f"\nSaving final batch of {len(results_batch)} records...")
        save_checkpoint(output_file, results_batch, append_mode)

    total_processed = len(processed_ids) + processed_count
    print(
        f"\nFinished '{input_file}': {total_processed} total "
        f"({processed_count} this run)."
    )


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    if args.temperature < 0.0 or args.temperature > 2.0:
        print(
            f"Warning: Temperature {args.temperature} is outside "
            "recommended range (0.0-2.0)"
        )

    client = make_client(args)

    try:
        io_pairs = resolve_io_pairs(args.input_file, args.output_file)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Will process {len(io_pairs)} input file(s):")
    for src, dst in io_pairs:
        print(f"  {src}  ->  {dst}")

    for src, dst in io_pairs:
        run_inference_on_file(client, src, dst, args)


if __name__ == "__main__":
    main()
