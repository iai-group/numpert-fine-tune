"""Sanity check: end-to-end data loading + balancing on real data.

Loads all perturbation files from ``data/perturbed_data/train_split/``,
runs them through the full pipeline (expand → deduplicate → convert →
balance), and prints a detailed report of:

  - Per-file and per-method raw counts
  - Deduplication effect
  - Perturbation-type distribution before and after balancing
  - Label distribution before and after balancing
  - Stratification by perturbation method (proportional vs equal)
  - Side-by-side comparison of proportional and equal-method modes

Usage::

    # Default: load all train_split files, no special balancing
    python -m tests.sanity_checks.check_data_balancing

    # Equal methods with class balancing
        python -m tests.sanity_checks.check_data_balancing \
            --equal_methods --class_ratio 1.0

    # Proportional, cap at 5000 samples
    python -m tests.sanity_checks.check_data_balancing \
        --max_samples 5000

    # Use val split instead
    python -m tests.sanity_checks.check_data_balancing --split val

    # Specific files only
    python -m tests.sanity_checks.check_data_balancing \
        --files numeration_True.json range_True.json

    # Compare proportional vs equal side-by-side
    python -m tests.sanity_checks.check_data_balancing --compare
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.llm_finetune.utils.data_loading.hf_dataset_instructions import (
    load_train_and_validation,
    FileLoadConfig,
)
from src.llm_finetune.utils.data_loading.data_balancer_old.method_balancer import MethodBalancer

SEP = "=" * 80
DATA_ROOT = Path("data/perturbed_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _print_counter(counter: Counter, total: int, col_width: int = 40) -> None:
    for key, count in counter.most_common():
        pct = 100 * count / total if total else 0
        print(f"    {str(key):<{col_width}}: {count:>5d}  ({pct:5.1f}%)")


def dataset_report(dataset, title: str = "Dataset") -> None:  # noqa: C901
    """Print a full breakdown of a HuggingFace Dataset."""
    _print_header(title)
    n = len(dataset)
    print(f"  Total examples: {n}")
    if n == 0:
        return

    cols = dataset.column_names
    print(f"  Columns: {cols}")

    # Perturbation type distribution
    print("\n  perturbation_type distribution:")
    pt_counter = Counter(dataset["perturbation_type"])
    _print_counter(pt_counter, n)

    # Label distribution (from conversations)
    labels = []
    for ex in dataset:
        last = ex["conversations"][-1]
        if last["role"] == "assistant":
            labels.append(last["content"])
        else:
            labels.append("(no label)")
    label_counter = Counter(labels)
    print("\n  Label distribution:")
    _print_counter(label_counter, n, col_width=20)

    # Perturbation method distribution (if column exists)
    if "perturbation_method" in cols:
        method_counter = Counter(dataset["perturbation_method"])
        print("\n  Perturbation method distribution:")
        _print_counter(method_counter, n)

    # Source file distribution (if column exists)
    if "source_file" in cols:
        sf_counter = Counter(dataset["source_file"])
        print("\n  Source file distribution:")
        _print_counter(sf_counter, n)

    # Cross-tab: perturbation method × perturbation_type
    if "perturbation_method" in cols:
        print("\n  Method × Perturbation type cross-tab:")
        cross: dict[str, Counter] = {}
        for ex in dataset:
            method = ex["perturbation_method"]
            ptype = ex["perturbation_type"]
            cross.setdefault(method, Counter())[ptype] += 1

        # Collect all types for header
        all_types = sorted(pt_counter.keys())
        header = f"    {'method':<20s}"
        for t in all_types:
            short = t.replace("perturb_", "p").replace(
                "original_normalized", "orig"
            )
            header += f"  {short:>8s}"
        header += f"  {'TOTAL':>8s}"
        print(header)
        print(f"    {'-' * (len(header) - 4)}")

        for method in sorted(cross):
            row = f"    {method:<20s}"
            total = 0
            for t in all_types:
                c = cross[method].get(t, 0)
                total += c
                row += f"  {c:>8d}"
            row += f"  {total:>8d}"
            print(row)


def balance_and_report(
    dataset,
    equal_methods: bool,
    class_ratio: float | None,
    max_samples: int | None,
    seed: int = 42,
    title: str = "Balanced",
):
    """Run MethodBalancer and print a report of the result."""
    balancer = MethodBalancer(
        dataset=dataset,
        equal_methods=equal_methods,
        class_ratio=class_ratio,
        max_samples=max_samples,
        include_originals=True,
        seed=seed,
    )
    balanced = balancer.balance()
    dataset_report(balanced, title)
    return balanced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main function to load, balance, and report on the dataset."""
    parser = argparse.ArgumentParser(
        description="Sanity check: data loading + balancing on real data",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to load (default: train)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            "Specific JSON filenames to load (e.g. numeration_True.json). "
            "Defaults to all files in the split directory."
        ),
    )
    parser.add_argument(
        "--equal_methods",
        action="store_true",
        help="Equalise sample count across perturbation methods",
    )
    parser.add_argument(
        "--class_ratio",
        type=float,
        default=None,
        help=(
            "Desired True:False ratio (e.g. 1.0 = 50/50). "
            "Omit to skip class balancing."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Hard cap on total output size (default: no cap)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both proportional AND equal-method and compare side-by-side",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    split_dir = DATA_ROOT / f"{args.split}_split"
    if not split_dir.exists():
        print(f"ERROR: {split_dir} does not exist.")
        return

    # Determine which files to load
    if args.files:
        json_files = [split_dir / f for f in args.files]
    else:
        json_files = sorted(split_dir.glob("*.json"))

    if not json_files:
        print(f"ERROR: No JSON files found in {split_dir}")
        return

    file_configs = [FileLoadConfig(path=str(f)) for f in json_files]

    _print_header(f"LOADING DATA FROM {split_dir}")
    print(f"  Files ({len(json_files)}):")
    for f in json_files:
        print(f"    {f.name}")

    # Load through the real pipeline
    datasets = load_train_and_validation(
        train_file_configs=file_configs,
        val_file_configs=file_configs[:1],  # need at least one for val
        expand_perturbations_flag=True,
        include_doc=True,
    )
    train_data = datasets["train"]

    # Report on the raw loaded data (pre-balancing)
    dataset_report(train_data, f"RAW LOADED DATA ({args.split}_split)")

    # Balance with the requested settings
    balance_and_report(
        dataset=train_data,
        equal_methods=args.equal_methods,
        class_ratio=args.class_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
        title=(
            f"BALANCED (equal_methods={args.equal_methods}, "
            f"class_ratio={args.class_ratio}, "
            f"max_samples={args.max_samples})"
        ),
    )

    # Optionally compare proportional vs equal
    if args.compare:
        _print_header("COMPARISON: PROPORTIONAL vs EQUAL-METHOD")

        balanced_prop = balance_and_report(
            dataset=train_data,
            equal_methods=False,
            class_ratio=args.class_ratio,
            max_samples=args.max_samples,
            seed=args.seed,
            title="PROPORTIONAL stratification",
        )

        balanced_equal = balance_and_report(
            dataset=train_data,
            equal_methods=True,
            class_ratio=args.class_ratio,
            max_samples=args.max_samples,
            seed=args.seed,
            title="EQUAL-METHOD stratification",
        )

        # Side-by-side method comparison
        if "perturbation_method" in train_data.column_names:
            print(f"\n{SEP}")
            print("  SIDE-BY-SIDE: method counts")
            print(SEP)

            prop_counts = Counter(balanced_prop["perturbation_method"])
            equal_counts = Counter(balanced_equal["perturbation_method"])
            all_methods = sorted(
                set(prop_counts.keys()) | set(equal_counts.keys())
            )

            prop_total = len(balanced_prop)
            equal_total = len(balanced_equal)

            header = (
                f"  {'method':<20s}"
                f"  {'proportional':>15s}"
                f"  {'equal-method':>15s}"
            )
            print(header)
            print(f"  {'-' * (len(header) - 2)}")

            for m in all_methods:
                pc = prop_counts.get(m, 0)
                ec = equal_counts.get(m, 0)
                pp = 100 * pc / prop_total if prop_total else 0
                ep = 100 * ec / equal_total if equal_total else 0
                print(
                    f"  {m:<20s}"
                    f"  {pc:>6d} ({pp:5.1f}%)"
                    f"  {ec:>6d} ({ep:5.1f}%)"
                )

            print(
                f"  {'TOTAL':<20s}"
                f"  {prop_total:>6d}"
                f"  {' ':>8s}"
                f"{equal_total:>6d}"
            )

    print(f"\n{SEP}")
    print("  DONE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
