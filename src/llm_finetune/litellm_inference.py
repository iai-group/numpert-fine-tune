"""Frontier model inference via litellm.

Runs fact-checking inference against any litellm-supported model (Anthropic,
OpenAI, etc.) using the same data pipeline and output format as
peft_inference.py so results are directly comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import litellm
import yaml
from tqdm import tqdm

from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.llm_finetune.utils.inference.inference_config_export import (
    build_inference_config,
    save_inference_config,
)
from src.llm_finetune.utils.inference.metrics import (
    compute_metrics,
    save_metrics_report,
)
from src.prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_PROMPT,
)

litellm.drop_params = True


def _load_litellm_config(config_path: str, model_name: str) -> dict:
    """Load per-model litellm params from a YAML config file.

    Resolves ``${ENV_VAR}`` placeholders from the environment.
    Returns a dict of kwargs to pass directly to ``litellm.completion()``.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    def _resolve(val: str) -> str:
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            val,
        )

    for entry in config.get("model_list", []):
        if entry.get("model_name") == model_name:
            params = entry.get("litellm_params", {})
            return {
                k: (_resolve(v) if isinstance(v, str) else v)
                for k, v in params.items()
            }

    raise ValueError(f"Model '{model_name}' not found in {config_path}")


def extract_label_from_response(response_text: str) -> Optional[bool]:
    """Extract boolean label from model response JSON."""
    if not response_text:
        return None
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return None
        parsed = json.loads(response_text[start_idx:end_idx])
        if "label" in parsed:
            label_val = parsed["label"]
            if isinstance(label_val, bool):
                return label_val
            if isinstance(label_val, str):
                return label_val.lower() == "true"
            return bool(label_val)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _call_model(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_params: dict | None = None,
) -> str:
    """Call litellm and return the response text."""
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **(extra_params or {}),
    )
    return response.choices[0].message.content or ""


def _infer_one(
    idx: int,
    example: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    extra_params: dict | None = None,
) -> tuple[dict, dict | None]:
    """Run inference on a single example.

    Returns a (prediction, parse_failure) tuple, where parse_failure is None
    when parsing succeeds.
    """
    label_true = example.get("label", None)

    # Build label JSON for storage
    if label_true is None:
        label_json = None
    elif isinstance(label_true, str) and label_true.strip().startswith("{"):
        label_json = label_true
    elif isinstance(label_true, str):
        label_json = json.dumps({"label": label_true.lower() == "true"})
    else:
        label_json = json.dumps({"label": bool(label_true)})

    # Build messages from claim + doc
    convs_raw = example.get("conversations")
    if convs_raw is not None:
        convs = (
            eval(convs_raw)
            if isinstance(convs_raw, str)
            else convs_raw  # noqa: S307
        )
        messages = [m for m in convs if m.get("role") in ("system", "user")]
        claim_text = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )[:200]
    else:
        claim = example.get("claim", "")
        doc = example.get("doc", "")
        claim_text = claim[:200]
        messages = [
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FACT_CHECK_USER_PROMPT.format(
                    examples="", claim=claim, evidence=doc
                ),
            },
        ]

    t0 = time.time()
    try:
        raw_response = _call_model(
            model, messages, max_tokens, temperature, extra_params
        )
        latency = time.time() - t0
        predicted_label = extract_label_from_response(raw_response)
        prediction_json = (
            json.dumps({"label": predicted_label})
            if predicted_label is not None
            else None
        )
        parse_failure = predicted_label is None
    except Exception as exc:
        raw_response = f"ERROR: {exc}"
        latency = time.time() - t0
        prediction_json = None
        parse_failure = True

    prediction = {
        "claim_id": example.get("claim_id", str(idx)),
        "claim": claim_text,
        "label": label_json,
        "prediction": prediction_json,
        "perturbation_type": example.get("perturbation_type", "unknown"),
        "source_file": example.get("source_file", ""),
        "raw_response": raw_response,
        "latency_seconds": round(latency, 3),
    }

    pf = None
    if parse_failure:
        pf = {
            "idx": idx,
            "claim_id": prediction["claim_id"],
            "perturbation_type": prediction["perturbation_type"],
            "source_file": prediction["source_file"],
            "label": label_json,
            "raw_response": raw_response,
        }

    return prediction, pf


def run_inference(
    model: str,
    test_json_path: str,
    output_dir: str = "./inference_results",
    max_workers: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
    sample_frac: float = 1.0,
    config_path: str | None = None,
) -> None:
    """Run inference using a litellm-supported frontier model.

    Args:
        model: litellm model string, e.g. ``"anthropic/claude-opus-4-7"``
            or ``"openai/gpt-4.5"``.
        test_json_path: Glob pattern for test JSON files.
        output_dir: Directory to save results.
        max_workers: Number of concurrent API threads.
        max_tokens: Maximum tokens in the model response.
        temperature: Sampling temperature (0.0 = deterministic).
        sample_frac: Fraction of test data to use (smoke-test helper).
        config_path: Optional path to YAML config with extra litellm params
    """
    extra_params: dict = {}
    if config_path:
        extra_params = _load_litellm_config(config_path, model)
        extra_params.pop("model", None)
        print(f"Config:    {config_path}")

    print(f"Model:     {model}")
    print(f"Test data: {test_json_path}")
    print(f"Workers:   {max_workers}")
    print()

    matched_files = sorted(glob.glob(test_json_path))
    if not matched_files:
        raise FileNotFoundError(f"No files matched: {test_json_path}")

    test_dataset = load_split(json_paths=matched_files, split="test")
    test_data = [test_dataset[i] for i in range(len(test_dataset))]

    if sample_frac < 1.0:
        n = max(1, math.ceil(len(test_data) * sample_frac))
        test_data = test_data[:n]
        print(f"Sampling {n}/{len(test_dataset)} examples ({sample_frac:.0%})")

    print(
        f"Loaded {len(test_data)} test examples from {len(matched_files)} file(s)\n"  # noqa: E501
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictions: list[dict | None] = [None] * len(test_data)
    parse_failures: list[dict] = []
    parse_fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _infer_one,
                idx,
                example,
                model,
                max_tokens,
                temperature,
                extra_params,
            ): idx
            for idx, example in enumerate(test_data)
        }

        with tqdm(
            total=len(test_data), unit="claim", dynamic_ncols=True
        ) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                prediction, pf = future.result()
                predictions[idx] = prediction
                if pf is not None:
                    parse_failures.append(pf)
                    parse_fail_count += 1
                pbar.update(1)
                pbar.set_postfix(parse_fails=parse_fail_count)

    # Metrics
    metrics_report = compute_metrics(predictions)

    # Save predictions
    with open(Path(output_dir) / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Save metrics
    save_metrics_report(metrics_report, output_dir)

    if parse_failures:
        with open(Path(output_dir) / "parse_failures.json", "w") as f:
            json.dump(parse_failures, f, indent=2)

    # Save CSV
    import pandas as pd

    pd.DataFrame(
        [
            {
                "claim_id": p["claim_id"],
                "perturbation_type": p.get("perturbation_type", ""),
                "source_file": p.get("source_file", ""),
                "label": p["label"],
                "prediction": p["prediction"]
                if p["prediction"] is not None
                else "PARSE_FAILURE",
                "parsed_successfully": p["prediction"] is not None,
                "raw_response": p.get("raw_response", ""),
            }
            for p in predictions
        ]
    ).to_csv(Path(output_dir) / "predictions.csv", index=False)

    # Save config
    config = build_inference_config(
        model_name=model,
        model_path=model,
        max_seq_length=max_tokens,
    )
    save_inference_config(config, output_dir)

    total = len(predictions)
    failures = sum(1 for p in predictions if p["prediction"] is None)
    print(f"\nDone. {total - failures}/{total} parsed successfully.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Frontier model inference via litellm"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="litellm model string, e.g. anthropic/claude-opus-4-7",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Glob pattern for test JSON files",
    )
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument(
        "--max_workers", type=int, default=10, help="Concurrent API threads"
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of test data (e.g. 0.05 for 5%% smoke test)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to litellm YAML config (e.g. configs/litellm_azure.yaml)",
    )

    args = parser.parse_args()
    run_inference(
        model=args.model,
        test_json_path=args.test_json,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sample_frac=args.sample_frac,
        config_path=args.config,
    )
