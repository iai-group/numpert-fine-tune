#!/usr/bin/env python3
"""Compare classification accuracy between two JSONL files.

Outputs custom IDs where file1 is correct but file2 is incorrect. Handles both
simple JSONL and OpenAI API response format.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class CustomIdParser:
    """Handles parsing of custom IDs to extract ground truth."""

    @staticmethod
    def parse(custom_id: str) -> Tuple[str, bool]:
        parts = custom_id.split("-")
        if len(parts) < 4:
            raise ValueError(f"Invalid custom_id format: {custom_id}")

        original_label = parts[1]
        perturbation_type = parts[-1].lower()
        category = "-".join(parts[2:-1])

        if perturbation_type == "baseline":
            if original_label.lower() == "false":
                gt = False
            elif original_label.lower() == "true":
                gt = True
            else:
                raise ValueError(
                    f"Unknown original label for baseline: {original_label}"
                )
        elif perturbation_type in ["t2t", "f2t"]:
            gt = True
        elif perturbation_type in ["t2f", "ft2f", "ff2f"]:
            gt = False
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        return category, gt


def extract_label_from_content(content: str) -> Optional[bool]:
    """Extract boolean label from string."""
    patterns = [
        r'\{\s*["\']?label["\']?\s*:\s*(true|false)\s*\}',
        r'["\']?label["\']?\s*:\s*(true|false)',
    ]
    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
    return None


def extract_label_from_openai_response(data: dict) -> Optional[bool]:
    try:
        choices = data.get("response", {}).get("body", {}).get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return extract_label_from_content(content)
    except Exception:
        return None


def load_jsonl_predictions(
    filepath: Path, verbose: bool = False
) -> Dict[str, bool]:
    """Load JSONL and map custom_id -> predicted_label (bool)."""
    preds = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Line {line_num}: JSON error {e}")
                continue

            if not isinstance(data, dict):
                if verbose:
                    print(f"Line {line_num}: not a dict, skipping: {data}")
                continue

            cid = data.get("custom_id") or data.get("id")
            if not cid:
                if verbose:
                    print(f"Line {line_num}: missing custom_id/id")
                continue

            pred = extract_label_from_openai_response(data)
            if pred is None:
                pred = extract_label_from_content(data.get("content", ""))

            if pred is None:
                if verbose:
                    print(f"Line {line_num}: could not extract label for {cid}")
                continue

            preds[cid] = pred
            if verbose and len(preds) <= 3:
                print(f"Debug: {cid} -> {pred}")

    print(f"Loaded {len(preds)} predictions from {filepath}")
    return preds


def evaluate_predictions(
    preds: Dict[str, bool]
) -> Dict[str, Tuple[bool, bool]]:
    """Return dict[cid] = (predicted_label, is_correct)."""
    out = {}
    for cid, pred in preds.items():
        try:
            _, gt = CustomIdParser.parse(cid)
            out[cid] = (pred, pred == gt)
        except ValueError as e:
            print(f"Skipping {cid}: {e}")
    return out


def compare_classifications(
    file1: Dict[str, Tuple[bool, bool]],
    file2: Dict[str, Tuple[bool, bool]],
    details: bool = False,
) -> List:
    """Return IDs (or full tuples) where file1 correct and file2 incorrect."""
    improved: list = []
    for cid in set(file1) & set(file2):
        pred1, corr1 = file1[cid]
        pred2, corr2 = file2[cid]
        if corr1 and not corr2:
            if details:
                _, gt = CustomIdParser.parse(cid)
                improved.append((cid, pred1, pred2, gt))
            else:
                improved.append(cid)
    return improved


def print_summary(
    file1: Dict[str, Tuple[bool, bool]],
    file2: Dict[str, Tuple[bool, bool]],
    improved: List,
):
    common = set(file1) & set(file2)
    f1_corr = sum(corr for _, corr in (file1[c] for c in common))
    f2_corr = sum(corr for _, corr in (file2[c] for c in common))
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Common IDs: {len(common)}")
    print(
        f"File1 accuracy: {f1_corr}/{len(common)} = {f1_corr/len(common):.2%}"
    )
    print(
        f"File2 accuracy: {f2_corr}/{len(common)} = {f2_corr/len(common):.2%}"
    )
    print(f"Improved (file1 correct, file2 wrong): {len(improved)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1", type=Path, help="First JSONL file")
    ap.add_argument("file2", type=Path, help="Second JSONL file")
    ap.add_argument("--details", action="store_true", help="Show preds + gt")
    ap.add_argument("--output", type=Path, help="Save improved IDs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    preds1 = load_jsonl_predictions(args.file1, verbose=args.verbose)
    preds2 = load_jsonl_predictions(args.file2, verbose=args.verbose)

    eval1 = evaluate_predictions(preds1)
    eval2 = evaluate_predictions(preds2)

    improved = compare_classifications(eval1, eval2, details=args.details)

    print_summary(eval1, eval2, improved)

    print("\n=== IMPROVED CUSTOM IDs (file1 correct, file2 incorrect) ===")
    if args.details:
        for cid, p1, p2, gt in improved:
            print(f"{cid} | file1={p1} | file2={p2} | gt={gt}")
    else:
        for cid in improved:
            print(cid)

    if args.output:
        with open(args.output, "w") as f:
            for cid in (
                cid if not args.details else cid[0] for cid in improved
            ):
                f.write(cid + "\n")
        print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
