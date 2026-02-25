#!/usr/bin/env python3
"""
Token Usage Analysis: Misclassifications vs Correct Predictions

This script analyzes JSONL files containing model predictions and calculates
the average token usage (total, completion, reasoning) for:
- Correctly classified samples  
- Misclassified samples

Ground truth is extracted from custom_id format: <ID-digit>-<groundtruth>-<type>...
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TokenMetrics:
    """Store token metrics for a group of predictions."""

    total_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]

    def __init__(self):
        self.total_tokens = []
        self.completion_tokens = []
        self.reasoning_tokens = []

    def add_sample(
        self,
        total: Optional[int],
        completion: Optional[int],
        reasoning: Optional[int],
    ):
        """Add a sample's token counts."""
        if total is not None:
            self.total_tokens.append(total)
        if completion is not None:
            self.completion_tokens.append(completion)
        if reasoning is not None:
            self.reasoning_tokens.append(reasoning)

    def get_stats(self, token_type: str) -> Dict[str, float]:
        """Calculate statistics for a specific token type."""
        if token_type == "total":
            values = self.total_tokens
        elif token_type == "completion":
            values = self.completion_tokens
        elif token_type == "reasoning":
            values = self.reasoning_tokens
        else:
            raise ValueError(f"Invalid token type: {token_type}")

        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0,
            }

        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "std": (
                round(statistics.stdev(values), 2) if len(values) > 1 else 0.0
            ),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }


class MisclassificationTokenAnalyzer:
    """Analyzes token usage patterns for correct vs incorrect predictions."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.correct_metrics = TokenMetrics()
        self.incorrect_metrics = TokenMetrics()
        self.processing_errors: list[str] = []

    def extract_ground_truth(self, custom_id: str) -> bool:
        """
        Extract ground truth from custom_id format: <ID-digit>-<groundtruth>-<type>...

        Args:
            custom_id: String in format "36-False-normalized_claim-baseline"

        Returns:
            Boolean ground truth value
        """
        parts = custom_id.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid custom_id format: {custom_id}")

        # Second part contains the ground truth
        truth_str = parts[1].lower()

        # Check if we need special handling for perturbation types
        if len(parts) >= 4:
            perturbation_type = parts[-1].lower()

            # Handle special perturbation cases
            if perturbation_type == "baseline":
                # For baseline, we need to consider the original label
                # In the original script, baseline inverts the label
                if truth_str == "false":
                    return True
                elif truth_str == "true":
                    return False
            elif perturbation_type in ["t2t", "f2t"]:
                return True
            elif perturbation_type in ["t2f", "ft2f", "ff2f"]:
                return False

        # Standard parsing
        if truth_str in ["true", "t", "1"]:
            return True
        elif truth_str in ["false", "f", "0"]:
            return False
        else:
            raise ValueError(
                f"Cannot parse ground truth '{parts[1]}' from custom_id: {custom_id}"
            )

    def extract_prediction(self, record: Dict) -> Optional[bool]:
        """Extract the model's prediction from the response.

        Args:
            record: Dictionary containing the response data

        Returns:
            Boolean prediction or None if extraction fails
        """
        try:
            # Navigate to the content
            response = record.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                return None

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return None

            # Method 1: Try parsing as JSON
            try:
                parsed = json.loads(content.strip())
                if isinstance(parsed, dict) and "label" in parsed:
                    label = parsed["label"]
                    if isinstance(label, bool):
                        return label
                    elif isinstance(label, str):
                        return label.lower() == "true"
            except json.JSONDecodeError:
                pass

            # Method 2: Use regex to find label
            patterns = [
                r'\{\s*["\']?label["\']?\s*:\s*(true|false)\s*\}',
                r'["\']?label["\']?\s*:\s*(true|false)',
                r'"label"\s*:\s*"?(true|false)"?',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    return matches[0].lower() == "true"

            return None

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to extract prediction: {e}")
            return None

    def extract_token_counts(
        self, record: Dict
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token counts from the response.

        Args:
            record: Dictionary containing the response data

        Returns:
            Tuple of (total_tokens, completion_tokens, reasoning_tokens)
        """
        try:
            response = record.get("response", {})
            body = response.get("body", {})
            usage = body.get("usage", {})

            total_tokens = usage.get("total_tokens")
            completion_tokens = usage.get("completion_tokens")

            # Reasoning tokens are in completion_tokens_details
            completion_details = usage.get("completion_tokens_details", {})
            reasoning_tokens = completion_details.get("reasoning_tokens")

            return total_tokens, completion_tokens, reasoning_tokens

        except Exception:
            return None, None, None

    def process_file(self, filepath: str) -> Dict:
        """Process a JSONL file and analyze token usage patterns.

        Args:
            filepath: Path to the JSONL file

        Returns:
            Dictionary containing analysis results
        """
        print(f"\n📖 Analyzing: {filepath}")

        # Reset metrics for new file
        self.correct_metrics = TokenMetrics()
        self.incorrect_metrics = TokenMetrics()
        self.processing_errors = []

        # Counters
        total_records = 0
        valid_predictions = 0
        correct_count = 0
        incorrect_count = 0
        missing_tokens = 0

        # Process file
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                total_records += 1

                try:
                    # Parse JSON record
                    record = json.loads(line)
                    custom_id = record.get("custom_id", "")

                    if not custom_id:
                        self.processing_errors.append(
                            f"Line {line_num}: Missing custom_id"
                        )
                        continue

                    # Extract ground truth
                    try:
                        ground_truth = self.extract_ground_truth(custom_id)
                    except ValueError as e:
                        self.processing_errors.append(
                            f"Line {line_num} ({custom_id}): {e}"
                        )
                        continue

                    # Extract prediction
                    prediction = self.extract_prediction(record)
                    if prediction is None:
                        self.processing_errors.append(
                            f"Line {line_num} ({custom_id}): Could not extract prediction"
                        )
                        continue

                    # Extract token counts
                    (
                        total_tokens,
                        completion_tokens,
                        reasoning_tokens,
                    ) = self.extract_token_counts(record)

                    if all(
                        t is None
                        for t in [
                            total_tokens,
                            completion_tokens,
                            reasoning_tokens,
                        ]
                    ):
                        missing_tokens += 1

                    # Determine if prediction is correct
                    is_correct = ground_truth == prediction
                    valid_predictions += 1

                    # Add to appropriate metrics
                    if is_correct:
                        self.correct_metrics.add_sample(
                            total_tokens, completion_tokens, reasoning_tokens
                        )
                        correct_count += 1
                    else:
                        self.incorrect_metrics.add_sample(
                            total_tokens, completion_tokens, reasoning_tokens
                        )
                        incorrect_count += 1

                    # Progress indicator
                    if line_num % 1000 == 0:
                        print(f"  Processed {line_num} records...")

                except json.JSONDecodeError:
                    self.processing_errors.append(
                        f"Line {line_num}: Invalid JSON"
                    )
                except Exception as e:
                    self.processing_errors.append(f"Line {line_num}: {e}")

        # Compile results
        results = {
            "file": str(filepath),
            "summary": {
                "total_records": total_records,
                "valid_predictions": valid_predictions,
                "correct_predictions": correct_count,
                "incorrect_predictions": incorrect_count,
                "accuracy": (
                    round(correct_count / valid_predictions * 100, 2)
                    if valid_predictions > 0
                    else 0
                ),
                "missing_token_records": missing_tokens,
                "processing_errors": len(self.processing_errors),
            },
            "token_metrics": {
                "correct_predictions": {
                    "total_tokens": self.correct_metrics.get_stats("total"),
                    "completion_tokens": self.correct_metrics.get_stats(
                        "completion"
                    ),
                    "reasoning_tokens": self.correct_metrics.get_stats(
                        "reasoning"
                    ),
                },
                "misclassifications": {
                    "total_tokens": self.incorrect_metrics.get_stats("total"),
                    "completion_tokens": self.incorrect_metrics.get_stats(
                        "completion"
                    ),
                    "reasoning_tokens": self.incorrect_metrics.get_stats(
                        "reasoning"
                    ),
                },
            },
            "comparison": self._calculate_comparison(),
        }

        # Add errors if verbose
        if self.verbose and self.processing_errors:
            results["sample_errors"] = self.processing_errors[:10]

        return results

    def _calculate_comparison(self) -> Dict:
        """Calculate comparison metrics between correct and incorrect
        predictions."""
        comparison = {}

        for token_type in ["total", "completion", "reasoning"]:
            correct_stats = self.correct_metrics.get_stats(token_type)
            incorrect_stats = self.incorrect_metrics.get_stats(token_type)

            if correct_stats["count"] > 0 and incorrect_stats["count"] > 0:
                diff = incorrect_stats["mean"] - correct_stats["mean"]
                pct_diff = (
                    (diff / correct_stats["mean"] * 100)
                    if correct_stats["mean"] > 0
                    else 0
                )

                comparison[f"{token_type}_tokens"] = {
                    "correct_mean": correct_stats["mean"],
                    "misclass_mean": incorrect_stats["mean"],
                    "absolute_difference": round(diff, 2),
                    "percentage_increase": round(pct_diff, 2),
                    "ratio": (
                        round(
                            incorrect_stats["mean"] / correct_stats["mean"], 3
                        )
                        if correct_stats["mean"] > 0
                        else 0
                    ),
                }
            else:
                comparison[f"{token_type}_tokens"] = {
                    "correct_mean": correct_stats["mean"],
                    "misclass_mean": incorrect_stats["mean"],
                    "absolute_difference": 0,
                    "percentage_increase": 0,
                    "ratio": 0,
                }

        return comparison

    def print_results(self, results: Dict):
        """Print formatted results to console."""
        print("\n" + "=" * 80)
        print(
            "🎯 TOKEN USAGE ANALYSIS: MISCLASSIFICATIONS VS CORRECT PREDICTIONS"
        )
        print("=" * 80)

        # Summary
        summary = results["summary"]
        print(f"\n📊 FILE SUMMARY")
        print(f"  Total Records: {summary['total_records']}")
        print(f"  Valid Predictions: {summary['valid_predictions']}")
        print(
            f"  ✅ Correct: {summary['correct_predictions']} ({summary['correct_predictions']/summary['valid_predictions']*100:.1f}%)"
            if summary["valid_predictions"] > 0
            else ""
        )
        print(
            f"  ❌ Incorrect: {summary['incorrect_predictions']} ({summary['incorrect_predictions']/summary['valid_predictions']*100:.1f}%)"
            if summary["valid_predictions"] > 0
            else ""
        )
        print(f"  Overall Accuracy: {summary['accuracy']}%")

        if summary["processing_errors"] > 0:
            print(f"  ⚠️  Processing Errors: {summary['processing_errors']}")

        # Token metrics for each type
        metrics = results["token_metrics"]
        comparison = results["comparison"]

        print("\n" + "=" * 80)
        print("📈 AVERAGE TOKEN USAGE COMPARISON")
        print("=" * 80)

        for token_type in ["total", "completion", "reasoning"]:
            print(f"\n🔹 {token_type.upper()} TOKENS")
            print("-" * 50)

            correct = metrics["correct_predictions"][f"{token_type}_tokens"]
            misclass = metrics["misclassifications"][f"{token_type}_tokens"]
            comp = comparison[f"{token_type}_tokens"]

            if correct["count"] > 0:
                print(f"  ✅ Correct Predictions (n={correct['count']}):")
                print(f"     Average: {correct['mean']:.1f} tokens")
                print(f"     Median:  {correct['median']:.1f} tokens")
                print(f"     Std Dev: {correct['std']:.1f} tokens")
                print(f"     Range:   [{correct['min']}, {correct['max']}]")
            else:
                print(f"  ✅ Correct Predictions: No data")

            if misclass["count"] > 0:
                print(f"  ❌ Misclassifications (n={misclass['count']}):")
                print(f"     Average: {misclass['mean']:.1f} tokens")
                print(f"     Median:  {misclass['median']:.1f} tokens")
                print(f"     Std Dev: {misclass['std']:.1f} tokens")
                print(f"     Range:   [{misclass['min']}, {misclass['max']}]")
            else:
                print(f"  ❌ Misclassifications: No data")

            # Comparison
            if correct["count"] > 0 and misclass["count"] > 0:
                print(f"\n  📊 Comparison:")
                if comp["percentage_increase"] > 0:
                    print(
                        f"     Misclassifications use {comp['absolute_difference']:.1f} MORE tokens"
                    )
                    print(
                        f"     This is a {comp['percentage_increase']:.1f}% increase"
                    )
                elif comp["percentage_increase"] < 0:
                    print(
                        f"     Misclassifications use {abs(comp['absolute_difference']):.1f} FEWER tokens"
                    )
                    print(
                        f"     This is a {abs(comp['percentage_increase']):.1f}% decrease"
                    )
                else:
                    print(f"     No difference in token usage")
                print(f"     Ratio (misclass/correct): {comp['ratio']:.2f}x")

        # Reasoning token percentage if available
        if metrics["correct_predictions"]["reasoning_tokens"]["count"] > 0:
            print("\n" + "=" * 80)
            print("💡 REASONING TOKEN INSIGHTS")
            print("=" * 80)

            correct_reasoning = metrics["correct_predictions"][
                "reasoning_tokens"
            ]["mean"]
            correct_completion = metrics["correct_predictions"][
                "completion_tokens"
            ]["mean"]

            if correct_completion > 0:
                correct_reasoning_pct = (
                    correct_reasoning / correct_completion
                ) * 100
                print(f"  Correct Predictions:")
                print(
                    f"    Reasoning tokens: {correct_reasoning_pct:.1f}% of completion tokens"
                )

            if metrics["misclassifications"]["reasoning_tokens"]["count"] > 0:
                misclass_reasoning = metrics["misclassifications"][
                    "reasoning_tokens"
                ]["mean"]
                misclass_completion = metrics["misclassifications"][
                    "completion_tokens"
                ]["mean"]

                if misclass_completion > 0:
                    misclass_reasoning_pct = (
                        misclass_reasoning / misclass_completion
                    ) * 100
                    print(f"  Misclassifications:")
                    print(
                        f"    Reasoning tokens: {misclass_reasoning_pct:.1f}% of completion tokens"
                    )

                    if correct_completion > 0:
                        print(
                            f"\n  Difference: {abs(misclass_reasoning_pct - correct_reasoning_pct):.1f} percentage points"
                        )


def process_directory(
    analyzer: MisclassificationTokenAnalyzer,
    directory: Path,
    output_file: Optional[Path] = None,
) -> List[Dict]:
    """Process all JSONL files in a directory."""
    jsonl_files = list(directory.glob("*.jsonl"))

    if not jsonl_files:
        print(f"❌ No JSONL files found in {directory}")
        return []

    print(f"\n📁 Found {len(jsonl_files)} JSONL files to process")

    all_results = []
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"\n{'='*40} File {i}/{len(jsonl_files)} {'='*40}")
        results = analyzer.process_file(str(file_path))
        analyzer.print_results(results)
        all_results.append(results)

    # Calculate aggregate statistics across all files
    if len(all_results) > 1:
        print_aggregate_stats(all_results)

    return all_results


def print_aggregate_stats(all_results: List[Dict]):
    """Print aggregate statistics across multiple files."""
    print("\n" + "=" * 80)
    print("📊 AGGREGATE STATISTICS ACROSS ALL FILES")
    print("=" * 80)

    total_correct = sum(
        r["summary"]["correct_predictions"] for r in all_results
    )
    total_incorrect = sum(
        r["summary"]["incorrect_predictions"] for r in all_results
    )
    total_valid = total_correct + total_incorrect

    print(f"\nTotal Samples: {total_valid}")
    print(
        f"  ✅ Correct: {total_correct} ({total_correct/total_valid*100:.1f}%)"
    )
    print(
        f"  ❌ Incorrect: {total_incorrect} ({total_incorrect/total_valid*100:.1f}%)"
    )

    # Aggregate token differences
    print("\nAverage Token Differences Across All Files:")

    for token_type in ["total", "completion", "reasoning"]:
        differences = []
        percentages = []

        for result in all_results:
            comp = result["comparison"][f"{token_type}_tokens"]
            if comp["correct_mean"] > 0 and comp["misclass_mean"] > 0:
                differences.append(comp["absolute_difference"])
                percentages.append(comp["percentage_increase"])

        if differences:
            avg_diff = statistics.mean(differences)
            avg_pct = statistics.mean(percentages)
            print(f"\n  {token_type.capitalize()} Tokens:")
            print(f"    Average difference: {avg_diff:.1f} tokens")
            print(f"    Average % increase: {avg_pct:.1f}%")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze token usage differences between correct predictions and misclassifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.jsonl
  %(prog)s results.jsonl -o analysis.json
  %(prog)s /path/to/directory --batch
  %(prog)s results.jsonl --verbose
        """,
    )

    parser.add_argument(
        "input", help="Path to JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "-o", "--output", help="Output JSON file to save results"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all JSONL files in the directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed error messages and debugging info",
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"❌ Error: '{args.input}' does not exist")
            sys.exit(1)

        # Initialize analyzer
        analyzer = MisclassificationTokenAnalyzer(verbose=args.verbose)

        # Process based on input type
        if input_path.is_file():
            if not str(input_path).endswith(".jsonl"):
                print("❌ Error: Input file must have .jsonl extension")
                sys.exit(1)

            # Process single file
            results = analyzer.process_file(str(input_path))
            analyzer.print_results(results)

            # Save results if output specified
            if args.output:
                output_path = Path(args.output)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {output_path}")

        elif input_path.is_dir():
            # Process directory
            all_results = process_directory(analyzer, input_path)

            # Save all results if output specified
            if args.output and all_results:
                output_path = Path(args.output)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                print(f"\n💾 All results saved to: {output_path}")

        else:
            print(f"❌ Error: '{args.input}' is neither a file nor a directory")
            sys.exit(1)

        print("\n✨ Analysis complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
