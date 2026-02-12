#!/usr/bin/env python3
"""Compare two JSONL model output files and create a nested JSON file that shows
the 'content' (model thoughts) for each custom_id.

The output filename is automatically generated from the two input files.
"""

import argparse
import json
from pathlib import Path
import code.error_analysis.compare_classifications_jsonl as cmp  # rename your original script to compare_script.py


def load_jsonl_raw(filepath: Path):
    """Load JSONL as raw dicts keyed by custom_id/id.

    Works with OpenAI batch JSONL (pretty-printed).
    """
    raw = {}
    buffer = ""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            buffer += line
            try:
                data = json.loads(buffer)
            except json.JSONDecodeError:
                # not complete JSON yet, keep buffering
                continue
            # got a full object
            cid = data.get("custom_id") or data.get("id")
            if cid:
                raw[cid] = data
            buffer = ""  # reset for next object
    if buffer.strip():
        raise ValueError(f"Trailing invalid JSON in {filepath}")
    return raw


def extract_fields(entry: dict) -> dict:
    """Extract both content and thoughts from a batch entry."""
    out = {"content": "", "thoughts": ""}
    try:
        choices = entry.get("response", {}).get("body", {}).get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            out["content"] = msg.get("content", "").strip()
            out["thoughts"] = msg.get("thoughts", "").strip()
            return out
    except Exception:
        pass
    # fallback if top-level
    out["content"] = entry.get("content", "").strip()
    out["thoughts"] = entry.get("thoughts", "").strip()
    return out


def build_comparison(file1: Path, file2: Path, output: Path):
    preds1 = cmp.load_jsonl_predictions(file1)
    preds2 = cmp.load_jsonl_predictions(file2)

    eval1 = cmp.evaluate_predictions(preds1)
    eval2 = cmp.evaluate_predictions(preds2)

    improved_ids = cmp.compare_classifications(eval1, eval2, details=True)

    raw1 = load_jsonl_raw(file1)
    raw2 = load_jsonl_raw(file2)

    name1 = file1.stem
    name2 = file2.stem

    results = {}
    for cid, p1, p2, gt in improved_ids:
        f1 = extract_fields(raw1.get(cid, {}))
        f2 = extract_fields(raw2.get(cid, {}))
        results[cid] = {
            "ground_truth": gt,
            name1: {
                "pred": p1,
                "content": f1["content"],
                "thoughts": f1["thoughts"],
            },
            name2: {
                "pred": p2,
                "content": f2["content"],
                "thoughts": f2["thoughts"],
            },
            "manual_notes": "",
        }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison JSON to {output}")


def make_output_name(file1: Path, file2: Path) -> Path:
    """Generate output filename based on input filenames."""
    base1 = file1.stem
    base2 = file2.stem
    return Path(f"{base1}_compared_{base2}_thoughts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1", type=Path, help="First JSONL file")
    ap.add_argument("file2", type=Path, help="Second JSONL file")
    args = ap.parse_args()

    output = make_output_name(args.file1, args.file2)
    build_comparison(args.file1, args.file2, output)


if __name__ == "__main__":
    main()
