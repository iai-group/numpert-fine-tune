"""Classification Score Metrics Calculator.

This script processes JSONL files containing model predictions and calculates
comprehensive classification metrics including accuracy, precision, recall, F1,
and confusion matrices both overall and per category.

Supports both single file processing and batch directory processing.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import re


class ResponseValidator:
    """Validates and extracts labels from model responses."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the validator with optional debug mode."""
        self.debug_mode = debug_mode
        self.debug_samples: List[Dict[str, Any]] = []

    def extract_label_from_json(self, content: str) -> Optional[bool]:
        """Try to extract label by parsing content as complete JSON."""
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "label" in parsed:
                label_value = parsed["label"]
                if isinstance(label_value, bool):
                    return label_value
                elif isinstance(label_value, str):
                    if label_value.lower() == "true":
                        return True
                    elif label_value.lower() == "false":
                        return False
            return None
        except (json.JSONDecodeError, KeyError, AttributeError):
            return None

    def extract_label_from_regex(self, content: str) -> List[str]:
        """Extract label values using regex patterns."""
        if not content:
            return []

        patterns = [
            r'\{\s*["\']?label["\']?\s*:\s*(true|false)\s*\}',
            r'["\']?label["\']?\s*:\s*(true|false)',
        ]

        all_matches: List[str] = []
        for pattern in patterns:
            all_matches.extend(re.findall(pattern, content, re.IGNORECASE))

        # Normalize and de-duplicate while preserving order
        seen = set()
        unique: List[str] = []
        for m in (match.lower() for match in all_matches):
            if m not in seen:
                seen.add(m)
                unique.append(m)

        return unique

    def validate_and_extract(self, content: str) -> Tuple[Optional[bool], str]:
        """Extract prediction from content and validate format.

        Returns:
            Tuple of (prediction, error_reason)
            - prediction: Boolean if valid, None if invalid
            - error_reason: Description of why invalid (empty if valid)
        """
        if not content or not content.strip():
            return None, "empty_content"

        content = content.strip()

        # First try complete JSON parsing
        json_result = self.extract_label_from_json(content)
        if json_result is not None:
            return json_result, ""

        # Then try regex extraction
        regex_matches = self.extract_label_from_regex(content)

        if len(regex_matches) == 0:
            if self.debug_mode and len(self.debug_samples) < 5:
                self.debug_samples.append(
                    f"No label found in: {content[:100]}..."
                )
            return None, "no_label_found"

        # Check for conflicting labels
        unique_matches = set(regex_matches)
        if len(unique_matches) > 1:
            return None, "conflicting_labels"

        # Single consistent label found
        label_str = regex_matches[0]
        return label_str == "true", ""

    def print_debug_info(self):
        """Print debug information about failed extractions."""
        if self.debug_samples:
            print("\n🔍 DEBUG: Sample failed extractions:")
            for i, sample in enumerate(self.debug_samples, 1):
                print(f"  {i}. {sample}")


class CustomIdParser:
    """Handles parsing of custom IDs to extract categories and ground truth."""

    @staticmethod
    def parse(custom_id: str) -> Tuple[str, bool]:
        """Parse custom_id to extract category and ground truth."""
        parts = custom_id.split("-")
        if len(parts) < 4:
            raise ValueError(
                f"Invalid custom_id format - expected at least 4 parts: {custom_id}"  # noqa: E501
            )

        original_label = parts[1]
        perturbation_type = parts[-1].lower()
        category_parts = parts[2:-1]
        category_base = "-".join(category_parts)

        # New category format: original_label-category-perturbation_type
        category = f"{original_label}-{category_base}-{perturbation_type}"

        # Determine ground truth based on perturbation type and original label
        if perturbation_type == "baseline":
            # For baseline cases, invert the original label
            if original_label.lower() == "false":
                ground_truth = True
            elif original_label.lower() == "true":
                ground_truth = False
            else:
                raise ValueError(
                    f"Unknown original label for baseline: {original_label}"
                )
        elif perturbation_type in ["t2t", "f2t"]:
            ground_truth = True
        elif perturbation_type in [
            "t2f",
            "ft2f",
            "ff2f",
        ]:  # Updated: ft2f and ff2f instead of f2f and f2ff
            ground_truth = False
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        return category, ground_truth


class JsonlProcessor:
    """Handles reading and processing JSONL files."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the processor with optional debug mode."""
        self.validator = ResponseValidator(debug_mode)
        self.parser = CustomIdParser()
        self.debug_mode = debug_mode
        self.failed_custom_ids: Dict[str, List[str]] = defaultdict(
            list
        )  # Track failed custom IDs by error type

    def extract_response_content(
        self, record: Dict
    ) -> Tuple[Optional[str], str]:
        """Extract content from nested response structure."""
        try:
            if "error" in record and record["error"] is not None:
                return None, "api_error"

            response = record.get("response", {})
            if not response:
                return None, "no_response"

            body = response.get("body", {})
            if not body:
                return None, "no_body"

            choices = body.get("choices", [])
            if not choices:
                return None, "no_choices"

            message = choices[0].get("message", {})
            if not message:
                return None, "no_message"

            content = message.get("content", "")
            if not content:
                return None, "empty_content"

            return content, ""

        except (KeyError, IndexError, AttributeError) as e:
            return None, f"structure_error: {str(e)}"

    def process_single_record(
        self, record: Dict, line_num: int
    ) -> Tuple[Optional[Dict], str]:
        """Process a single JSONL record."""
        try:
            # Extract custom_id
            custom_id = record.get("custom_id", "")
            if not custom_id:
                return None, "missing_custom_id"

            # Parse custom_id
            try:
                category, ground_truth = self.parser.parse(custom_id)
            except Exception as e:
                error_type = "custom_id_parse_error"
                self.failed_custom_ids[error_type].append(custom_id)
                return None, f"{error_type}: {str(e)}"

            # Extract response content
            content, content_error = self.extract_response_content(record)
            if content_error:
                self.failed_custom_ids[content_error].append(custom_id)
                return None, content_error

            # Validate and extract prediction
            prediction, validation_error = self.validator.validate_and_extract(
                content
            )
            if validation_error:
                self.failed_custom_ids[validation_error].append(custom_id)
                return None, validation_error

            return {
                "custom_id": custom_id,
                "category": category,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "line_num": line_num,
                "content_sample": (
                    content[:100] + "..." if len(content) > 100 else content
                ),
            }, ""

        except Exception as e:
            error_type = "unexpected_error"
            custom_id = record.get("custom_id", f"line_{line_num}")
            self.failed_custom_ids[error_type].append(custom_id)
            return None, f"{error_type}: {str(e)}"

    def process_file(  # noqa: C901
        self, filepath: str
    ) -> Tuple[List[Dict], Dict[str, int], Dict[str, List[str]]]:
        """Process entire JSONL file."""
        processed_data = []
        error_counts: Dict[str, int] = defaultdict(int)
        self.failed_custom_ids = defaultdict(list)  # Reset for each file

        print(f"📖 Reading file: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            raise Exception(f"Failed to read file: {str(e)}")

        print(f"📄 Found {len(lines)} lines to process")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                error_counts["json_decode_error"] += 1
                custom_id = f"line_{line_num}"
                self.failed_custom_ids["json_decode_error"].append(custom_id)
                if self.debug_mode and error_counts["json_decode_error"] <= 3:
                    print(
                        "  DEBUG: JSON decode error"
                        f" on line {line_num}: {str(e)}"
                    )
                continue

            # Process record
            result, error = self.process_single_record(record, line_num)
            if result:
                processed_data.append(result)
            else:
                error_counts[
                    error.split(":")[0]
                ] += 1  # Use base error type for counting

            # Progress indicator for large files
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines...")

        # Print debug info
        if self.debug_mode:
            self.validator.print_debug_info()
            print("\n📊 Processing summary:")
            print(f"  Valid records: {len(processed_data)}")
            print("  Error breakdown:")
            for error_type, count in error_counts.items():
                print(f"    {error_type}: {count}")
                if len(self.failed_custom_ids[error_type]) <= 10:
                    print(
                        f"      Failed IDs: {self.failed_custom_ids[error_type]}"  # noqa: E501
                    )
                else:
                    print(
                        f"      Failed IDs (first 10): {self.failed_custom_ids[error_type][:10]}"  # noqa: E501
                    )

        return processed_data, dict(error_counts), dict(self.failed_custom_ids)


class MetricsCalculator:
    """Handles calculation of classification metrics."""

    @staticmethod
    def calculate_binary_metrics(
        tp: int, tn: int, fp: int, fn: int
    ) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        total = tp + tn + fp + fn

        if total == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "specificity": 0.0,
                "npv": 0.0,
            }

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * (precision * recall) / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "npv": npv,
        }

    def calculate_confusion_matrix(
        self, data: List[Dict]
    ) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix values."""
        tp = tn = fp = fn = 0

        for record in data:
            gt, pred = record["ground_truth"], record["prediction"]
            if gt and pred:
                tp += 1
            elif not gt and not pred:
                tn += 1
            elif not gt and pred:
                fp += 1
            else:  # gt and not pred
                fn += 1

        return tp, tn, fp, fn

    def calculate_all_metrics(
        self,
        data: List[Dict],
        error_counts: Dict[str, int],
        failed_ids: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        if not data:
            raise ValueError("No valid data to process")

        # Group data by category
        categories = defaultdict(list)
        for record in data:
            categories[record["category"]].append(record)

        # Calculate overall metrics
        tp, tn, fp, fn = self.calculate_confusion_matrix(data)
        overall_metrics = self.calculate_binary_metrics(tp, tn, fp, fn)
        overall_confusion = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": len(data),
        }

        # Calculate per-category metrics
        category_metrics = {}
        for category, records in categories.items():
            tp, tn, fp, fn = self.calculate_confusion_matrix(records)
            metrics = self.calculate_binary_metrics(tp, tn, fp, fn)
            confusion = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total": len(records),
            }

            category_metrics[category] = {
                "metrics": metrics,
                "confusion_matrix": confusion,
                "sample_count": len(records),
            }

        # Calculate totals
        total_invalid = sum(error_counts.values())
        total_processed = len(data) + total_invalid

        return {
            "overall": {
                "metrics": overall_metrics,
                "confusion_matrix": overall_confusion,
            },
            "categories": category_metrics,
            "summary": {
                "total_samples": len(data),
                "total_categories": len(categories),
                "categories_list": sorted(categories.keys()),
                "total_processed": total_processed,
                "valid_responses": len(data),
                "invalid_responses": total_invalid,
                "error_breakdown": error_counts,
                "failed_custom_ids": failed_ids,
                "valid_rate": (
                    len(data) / total_processed if total_processed > 0 else 0.0
                ),
            },
        }


class ResultsFormatter:
    """Handles formatting and display of results."""

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format a decimal as percentage with 2 decimal places."""
        return f"{value * 100:.2f}%"

    def print_results(
        self, results: Dict[str, Any], filename: str = ""
    ) -> None:
        """Print formatted results to console."""
        title = "🎯 CLASSIFICATION METRICS RESULTS"
        if filename:
            title += f" - {filename}"

        print(title)
        print("=" * 60)

        # Summary with error tracking
        summary = results["summary"]
        print("\n📊 SUMMARY")
        print(f"Total Processed: {summary['total_processed']}")
        print(
            f"Valid Responses: {summary['valid_responses']}"
            f" ({self.format_percentage(summary['valid_rate'])})"
        )
        print(f"Invalid Responses: {summary['invalid_responses']}")

        # Error breakdown with sample failed IDs
        if summary["invalid_responses"] > 0:
            print("\n❌ ERROR BREAKDOWN")
            failed_ids = summary.get("failed_custom_ids", {})
            for error_type, count in summary["error_breakdown"].items():
                if count > 0:
                    error_name = error_type.replace("_", " ").title()
                    print(f"  {error_name}: {count}")

                    # Show sample failed custom IDs
                    if error_type in failed_ids and failed_ids[error_type]:
                        sample_ids = failed_ids[error_type][:5]  # Show first 5
                        print(f"    Sample failed IDs: {sample_ids}")
                        if len(failed_ids[error_type]) > 5:
                            print(
                                f"    ... and {len(failed_ids[error_type]) - 5} more"  # noqa: E501
                            )

        if summary["valid_responses"] == 0:
            print("\n⚠️  No valid responses found - cannot calculate metrics")
            return

        print("\n📈 CLASSIFICATION ANALYSIS (Valid Responses Only)")
        print(f"Categories: {', '.join(summary['categories_list'])}")

        # Overall metrics
        print("\n🔍 OVERALL METRICS")
        overall = results["overall"]
        metrics = overall["metrics"]
        confusion = overall["confusion_matrix"]

        print(f"Accuracy:    {self.format_percentage(metrics['accuracy'])}")
        print(f"Precision:   {self.format_percentage(metrics['precision'])}")
        print(f"Recall:      {self.format_percentage(metrics['recall'])}")
        print(f"F1 Score:    {self.format_percentage(metrics['f1'])}")

        print("\n📋 CONFUSION MATRIX")
        print(f"{'':>12} {'Predicted':>20}")
        print(f"{'':>12} {'True':>8} {'False':>8}")
        print(f"{'Actual True':>12} {confusion['tp']:>8} {confusion['fn']:>8}")
        print(f"{'Actual False':>12} {confusion['fp']:>8} {confusion['tn']:>8}")

        # Category metrics
        if len(results["categories"]) > 1:
            print("\n📂 PER-CATEGORY METRICS")
            print("-" * 60)

            for category, cat_data in sorted(results["categories"].items()):
                metrics = cat_data["metrics"]
                confusion = cat_data["confusion_matrix"]

                print(
                    f"\n📁 {category.replace('_', ' ')}"
                    f" ({cat_data['sample_count']} samples)"
                )
                print(
                    f"  Accuracy: {self.format_percentage(metrics['accuracy']):>6}"  # noqa: E501
                    f"  |  F1: {self.format_percentage(metrics['f1']):>6}"
                )
                print(
                    f"  Precision: {self.format_percentage(metrics['precision']):>5}"  # noqa: E501
                    f"  |  Recall: {self.format_percentage(metrics['recall']):>5}"  # noqa: E501
                )

    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str) -> None:
        """Save results to JSON file."""
        # Remove content_sample from data before saving (too verbose)
        clean_results = json.loads(json.dumps(results))  # Deep copy

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {output_path}")


class ClassificationMetricsRunner:
    """Main runner class that orchestrates the entire process."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the runner with optional debug mode."""
        self.debug_mode = debug_mode
        self.processor = JsonlProcessor(debug_mode)
        self.calculator = MetricsCalculator()
        self.formatter = ResultsFormatter()

    def get_output_filename(
        self, input_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """Generate output filename with RESULTS_ prefix."""
        input_path_obj: Path = Path(input_path)

        if output_dir:
            output_dir_obj: Path = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            return str(output_dir_obj / f"RESULTS_{input_path_obj.stem}.json")
        else:
            return str(input_path_obj.parent / f"RESULTS_{input_path_obj.stem}.json")

    def find_jsonl_files(self, directory: Union[str, Path]) -> List[str]:
        """Find all JSONL files in a directory."""
        directory_path: Path = Path(directory)
        jsonl_files: List[str] = []

        for file_path in directory_path.rglob("*.jsonl"):
            jsonl_files.append(str(file_path))

        return sorted(jsonl_files)

    def process_single_file(
        self, input_file: str, output_path: str = None
    ) -> None:
        """Process a single JSONL file."""
        print(f"🔄 Processing file: {input_file}")

        try:
            # Process JSONL file
            data, error_counts, failed_ids = self.processor.process_file(
                input_file
            )

            if not data:
                print("❌ No valid data found in file")
                if sum(error_counts.values()) > 0:
                    print("📊 Error summary:")
                    for error_type, count in error_counts.items():
                        if count > 0:
                            print(f"  {error_type}: {count}")
                            # Show sample failed IDs
                            if error_type in failed_ids:
                                sample_ids = failed_ids[error_type][:3]
                                print(f"    Sample failed IDs: {sample_ids}")
                return

            print(f"✅ Processed {len(data)} valid records")

            # Calculate metrics
            results = self.calculator.calculate_all_metrics(
                data, error_counts, failed_ids
            )

            # Print results
            self.formatter.print_results(results, Path(input_file).name)

            # Save results
            if not output_path:
                output_path = self.get_output_filename(input_file)

            self.formatter.save_results(results, output_path)

        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            if self.debug_mode:
                import traceback

                traceback.print_exc()

    def process_directory(self, input_dir: str, output_dir: str = None) -> None:
        """Process all JSONL files in a directory."""
        print(f"🔄 Processing directory: {input_dir}")

        jsonl_files = self.find_jsonl_files(input_dir)

        if not jsonl_files:
            print("❌ No JSONL files found in directory")
            return

        print(f"📁 Found {len(jsonl_files)} JSONL files")

        for i, file_path in enumerate(jsonl_files, 1):
            print(f"\n{'='*20} File {i}/{len(jsonl_files)} {'='*20}")

            output_path = (
                self.get_output_filename(file_path, output_dir)
                if output_dir
                else None
            )
            self.process_single_file(file_path, output_path)

        print("\n🎉 Directory processing complete!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Calculate classification metrics from JSONL file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to input JSONL file or directory")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output JSON file or directory (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed error info",
    )

    args = parser.parse_args()

    try:
        runner = ClassificationMetricsRunner(debug_mode=args.debug)
        input_path = Path(args.input)

        if input_path.is_file():
            if not input_path.suffix.lower() == ".jsonl":
                print("❌ Error: Input file must have .jsonl extension")
                sys.exit(1)
            runner.process_single_file(str(input_path), args.output)
        elif input_path.is_dir():
            runner.process_directory(str(input_path), args.output)
        else:
            print(f"❌ Error: '{args.input}' is not a valid file or directory")
            sys.exit(1)

    except FileNotFoundError:
        print(f"❌ Error: '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
