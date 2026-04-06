"""Sanity check: dataset balancer on real perturbed data."""


from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.llm_finetune.utils.data_loading.dataset_balancer import (
    Sample,
    build_balanced_dataset,
    compute_stats,
    deduplicate,
    flatten_samples,
    load_files,
)

SEP = "=" * 80
THIN_SEP = "-" * 80
DATA_ROOT = Path("data/perturbed_data")


# -------------------------------------------------------------------
# Printing helpers
# -------------------------------------------------------------------


def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _subheader(title: str) -> None:
    print(f"\n  {THIN_SEP}")
    print(f"  {title}")
    print(f"  {THIN_SEP}")


def _counter_table(
    counter: Counter,
    total: int,
    col: int = 35,
) -> None:
    for key, count in counter.most_common():
        pct = 100 * count / total if total else 0
        print(f"    {str(key):<{col}}: {count:>6d}  ({pct:5.1f}%)")


# -------------------------------------------------------------------
# Report helpers
# -------------------------------------------------------------------


def report_raw_input(
    file_data: list[tuple[str, list[dict]]],
) -> None:
    """Print per-file record counts and label/method breakdowns."""
    _header("RAW INPUT")

    total_records = 0
    label_ctr: Counter = Counter()
    method_ctr: Counter = Counter()

    print(f"  {'File':<35s}  {'Records':>8s}  {'True':>6s}  {'False':>6s}")
    print(f"  {THIN_SEP}")

    for fname, records in file_data:
        n = len(records)
        total_records += n
        t = sum(1 for r in records if r.get("label") == "True")
        f = n - t
        label_ctr["True"] += t
        label_ctr["False"] += f
        for r in records:
            m = r.get("perturbation_method", "unknown")
            method_ctr[m] += 1
        print(f"  {fname:<35s}  {n:>8d}  {t:>6d}  {f:>6d}")

    print(f"  {THIN_SEP}")
    print(
        f"  {'TOTAL':<35s}  {total_records:>8d}  "
        f"{label_ctr['True']:>6d}  {label_ctr['False']:>6d}"
    )

    _subheader("Perturbation method (from raw records)")
    _counter_table(method_ctr, total_records)


def report_flattened(samples: list[Sample]) -> None:
    """Print breakdown of flattened samples."""
    _header("FLATTENED SAMPLES")

    n = len(samples)
    originals = [s for s in samples if not s.is_perturbed]
    perturbed = [s for s in samples if s.is_perturbed]

    print(f"  Total flattened samples : {n}")
    print(f"  Originals               : {len(originals)}")
    print(f"  Perturbations           : {len(perturbed)}")

    _subheader("Label distribution")
    _counter_table(Counter(s.label for s in samples), n)

    _subheader("Type distribution")
    _counter_table(Counter(s.type for s in samples), n)

    _subheader("Method distribution (perturbed only)")
    method_ctr = Counter(s.perturbation_method for s in perturbed)
    _counter_table(method_ctr, len(perturbed))

    _subheader("Source file distribution")
    _counter_table(Counter(s.source_file for s in samples), n)


def report_dedup(before: list[Sample], after: list[Sample]) -> None:
    """Print deduplication effect."""
    _header("DEDUPLICATION")

    removed = len(before) - len(after)
    print(f"  Before : {len(before)}")
    print(f"  After  : {len(after)}")
    print(f"  Removed: {removed}")

    if removed == 0:
        print("  (no duplicates found)")
        return

    # Break down removals by type
    before_types = Counter(s.type for s in before)
    after_types = Counter(s.type for s in after)
    all_types = sorted(set(before_types.keys()) | set(after_types.keys()))

    print(f"\n  {'Type':<15s}  {'Before':>8s}  {'After':>8s}  {'Removed':>8s}")
    print(f"  {THIN_SEP}")
    for t in all_types:
        b = before_types.get(t, 0)
        a = after_types.get(t, 0)
        print(f"  {t:<15s}  {b:>8d}  {a:>8d}  {b - a:>8d}")


def report_balanced(result: list[dict], title: str = "BALANCED OUTPUT") -> None:
    """Print full breakdown of the balanced result."""
    _header(title)

    stats = compute_stats(result)
    n = stats["total_samples"]
    print(f"  Total samples: {n}")

    _subheader("Label distribution")
    _counter_table(Counter(stats["label_distribution"]), n, col=15)

    _subheader("Type distribution")
    _counter_table(Counter(stats["type_distribution"]), n)

    _subheader("Perturbation method distribution")
    _counter_table(Counter(stats["method_distribution"]), n)

    _subheader("Source file distribution")
    _counter_table(Counter(stats["file_distribution"]), n)

    # Originals vs perturbed
    n_orig = sum(1 for s in result if not s["is_perturbed"])
    n_pert = sum(1 for s in result if s["is_perturbed"])
    actual_ratio = n_pert / n_orig if n_orig else float("inf")
    print(f"\n  Originals : {n_orig}")
    print(f"  Perturbed : {n_pert}")
    print(f"  Ratio (pert/orig): {actual_ratio:.2f}")

    t_to_t_total = sum(
        1 for s in result if s["is_perturbed"] and s["type"] == "T→T"
    )
    false_pert_total = sum(
        1 for s in result if s["is_perturbed"] and s["type"] in {"T→F", "F→F"}
    )
    if n_pert:
        print("\n  Perturbation class composition:")
        print(
            f"    T→T       : {t_to_t_total:>5d}  "
            f"({100 * t_to_t_total / n_pert:.1f}%)"
        )
        print(
            f"    False pert: {false_pert_total:>5d}  "
            f"({100 * false_pert_total / n_pert:.1f}%)"
        )

    # T→F / F→F breakdown among False perturbations
    false_pert = [
        s for s in result if s["is_perturbed"] and s["label"] == "False"
    ]
    if false_pert:
        t_to_f = sum(1 for s in false_pert if s["type"] == "T→F")
        f_to_f = sum(1 for s in false_pert if s["type"] == "F→F")
        total_fp = t_to_f + f_to_f
        ratio = t_to_f / total_fp if total_fp else 0
        print("\n  False perturbation composition:")
        print(f"    T→F : {t_to_f:>5d}  ({100 * ratio:.1f}%)")
        print(f"    F→F : {f_to_f:>5d}  ({100 * (1 - ratio):.1f}%)")

    # Cross-tab: method × type
    _subheader("Method × Type cross-tab")
    cross: dict[str, Counter] = defaultdict(Counter)
    for s in result:
        m = s["perturbation_method"] or "(original)"
        cross[m][s["type"]] += 1

    all_types = sorted({s["type"] for s in result})
    hdr = f"  {'method':<20s}"
    for t in all_types:
        hdr += f"  {t:>10s}"
    hdr += f"  {'TOTAL':>8s}"
    print(hdr)
    print(f"  {THIN_SEP}")
    for method in sorted(cross):
        row = f"  {method:<20s}"
        total = 0
        for t in all_types:
            c = cross[method].get(t, 0)
            total += c
            row += f"  {c:>10d}"
        row += f"  {total:>8d}"
        print(row)


def report_comparison(
    configs: list[dict],
    file_paths: list[str | Path],
) -> None:
    """Run multiple configs and print a side-by-side summary."""
    _header("MULTI-CONFIG COMPARISON")

    results = {}
    for cfg in configs:
        name = cfg.pop("name")
        result = build_balanced_dataset(file_paths=file_paths, **cfg)
        results[name] = result
        cfg["name"] = name  # restore

    # Summary table
    print(
        f"\n  {'Config':<30s}  {'Total':>6s}  {'True':>6s}  "
        f"{'False':>6s}  {'Orig':>6s}  {'Pert':>6s}  "
        f"{'P/O':>5s}  {'T→T':>6s}  {'T→F':>6s}  "
        f"{'F→F':>6s}  {'Methods':>8s}"
    )
    print(f"  {THIN_SEP}")

    for name, result in results.items():
        n = len(result)
        t = sum(1 for s in result if s["label"] == "True")
        f = n - t
        orig = sum(1 for s in result if not s["is_perturbed"])
        pert = n - orig
        po = pert / orig if orig else 0
        perturbed = [s for s in result if s["is_perturbed"]]
        t_to_t = sum(1 for s in perturbed if s["type"] == "T→T")
        t_to_f = sum(1 for s in perturbed if s["type"] == "T→F")
        f_to_f = sum(1 for s in perturbed if s["type"] == "F→F")
        methods = len(
            {
                s["perturbation_method"]
                for s in result
                if s["perturbation_method"] is not None
            }
        )
        print(
            f"  {name:<30s}  {n:>6d}  {t:>6d}  "
            f"{f:>6d}  {orig:>6d}  {pert:>6d}  "
            f"{po:>5.2f}  {t_to_t:>6d}  {t_to_f:>6d}  "
            f"{f_to_f:>6d}  {methods:>8d}"
        )

    # Per-config method breakdown
    for name, result in results.items():
        _subheader(f"{name}: method counts")
        method_ctr = Counter(
            s["perturbation_method"] or "(original)" for s in result
        )
        _counter_table(method_ctr, len(result))


def write_output_json(output_path: Path, result: list[dict]) -> None:
    """Write the sampled balanced dataset to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------
# Preset configurations
# -------------------------------------------------------------------

PRESET_CONFIGS = [
    {
        "name": "defaults",
        "balance_labels": True,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "none",
        "seed": 42,
    },
    {
        "name": "no_balancing",
        "balance_labels": False,
        "perturb_ratio": 100.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": None,
        "equal_method_size": False,
        "method_balance_mode": "none",
        "seed": 42,
    },
    {
        "name": "fair_split+oversample",
        "balance_labels": False,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "fair_split+undersample",
        "balance_labels": False,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": True,
        "method_balance_mode": "undersample",
        "seed": 42,
    },
    {
        "name": "t_to_t_heavy (50%)",
        "balance_labels": False,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 0.5,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "false_heavy (20% T→T)",
        "balance_labels": False,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 0.2,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "ratio_0.5+fair_split",
        "balance_labels": False,
        "perturb_ratio": 0.5,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "ratio_2.0+fair_split",
        "balance_labels": False,
        "perturb_ratio": 2.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "ratio_2.0+t_to_t_heavy",
        "balance_labels": False,
        "perturb_ratio": 2.0,
        "true_perturbation_fraction": 0.5,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "ratio_2.0+false_heavy",
        "balance_labels": False,
        "perturb_ratio": 2.0,
        "true_perturbation_fraction": 0.2,
        "false_perturb_t_to_f_ratio": 0.5,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "t_to_f_heavy (80%)",
        "balance_labels": True,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.8,
        "equal_method_size": False,
        "method_balance_mode": "none",
        "seed": 42,
    },
    {
        "name": "f_to_f_heavy (20%)",
        "balance_labels": True,
        "perturb_ratio": 1.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.2,
        "equal_method_size": False,
        "method_balance_mode": "none",
        "seed": 42,
    },
    {
        "name": "ratio_2.0+t_to_f_heavy",
        "balance_labels": False,
        "perturb_ratio": 2.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.8,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
    {
        "name": "ratio_2.0+f_to_f_heavy",
        "balance_labels": False,
        "perturb_ratio": 2.0,
        "true_perturbation_fraction": 1 / 3,
        "false_perturb_t_to_f_ratio": 0.2,
        "equal_method_size": False,
        "method_balance_mode": "oversample",
        "seed": 42,
    },
]


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:  # noqa: C901
    r"""Load perturbation files from ``data/perturbed_data/{split}_split/``.

    Runs them through the dataset_balancer pipeline and prints a detailed
    report at every stage:

    - Raw input counts per file and per method
    - Flattened sample counts (originals vs perturbations)
    - Deduplication effect
    - Balanced output with label / type / method / file distributions
    - Multiple configuration runs for easy comparison

    Usage::

        # Default: all train_split files, default balancing
        python -m tests.sanity_checks.check_dataset_balancer

        # Use test split
        python -m tests.sanity_checks.check_dataset_balancer --split test

        # Specific files only
        python -m tests.sanity_checks.check_dataset_balancer \\
            --files numeration_True.json range_True.json

        # Custom single run
        python -m tests.sanity_checks.check_dataset_balancer \\
            --perturb_ratio 2.0 --true_perturbation_fraction 0.33 \
            --method_balance_mode oversample --balance_labels

        # Write the sampled balanced dataset to JSON
        python -m tests.sanity_checks.check_dataset_balancer \
            --output_json results/balanced_dataset_debug.json

        # Run all preset configurations for comparison
        python -m tests.sanity_checks.check_dataset_balancer --compare
    """
    parser = argparse.ArgumentParser(
        description="Sanity check: dataset balancer on real data",
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
        help="Specific JSON filenames (e.g. numeration_True.json).",
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        default=False,
        help="Enforce equal True/False counts.",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=1.0,
        help="Perturbed/original ratio (default: 1.0).",
    )
    parser.add_argument(
        "--false_perturb_t_to_f_ratio",
        type=float,
        default=0.5,
        help=(
            "Share of False perturbations that should be T→F " "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--true_perturbation_fraction",
        type=float,
        default=1 / 3,
        help=(
            "Share of all perturbed examples that should remain True "
            "(T→T). Default 0.333 gives a fair T→T/T→F/F→F split "
            "when false_perturb_t_to_f_ratio=0.5."
        ),
    )
    parser.add_argument(
        "--equal_methods",
        action="store_true",
        default=False,
        help="Backward-compatible alias for --method_balance_mode undersample.",
    )
    parser.add_argument(
        "--method_balance_mode",
        choices=["none", "undersample", "oversample"],
        default=None,
        help=(
            "How to balance perturbation methods: leave unchanged, "
            "undersample to the smallest method, or oversample to the largest."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all preset configurations and compare.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help=(
            "Optional path to write the sampled balanced dataset from the "
            "single-run configuration as JSON."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    split_dir = DATA_ROOT / f"{args.split}_split"
    if not split_dir.exists():
        print(f"ERROR: {split_dir} does not exist.")
        return

    if args.files:
        json_files = [split_dir / f for f in args.files]
    else:
        json_files = sorted(split_dir.glob("*.json"))

    if not json_files:
        print(f"ERROR: No JSON files found in {split_dir}")
        return

    _header(f"DATASET BALANCER SANITY CHECK — {split_dir}")
    print(f"  Files ({len(json_files)}):")
    for f in json_files:
        print(f"    {f.name}")

    # ------- Stage 1: raw input -------
    file_data = load_files(json_files)
    report_raw_input(file_data)

    # ------- Stage 2: flatten -------
    flat = flatten_samples(file_data)
    report_flattened(flat)

    # ------- Stage 3: deduplication -------
    deduped = deduplicate(flat)
    report_dedup(flat, deduped)

    # ------- Stage 4: single balanced run -------
    result = build_balanced_dataset(
        file_paths=json_files,
        balance_labels=args.balance_labels,
        perturb_ratio=args.perturb_ratio,
        false_perturb_t_to_f_ratio=args.false_perturb_t_to_f_ratio,
        true_perturbation_fraction=args.true_perturbation_fraction,
        equal_method_size=args.equal_methods,
        method_balance_mode=args.method_balance_mode,
        seed=args.seed,
    )
    report_balanced(
        result,
        title=(
            f"BALANCED OUTPUT  "
            f"(labels={args.balance_labels}, "
            f"ratio={args.perturb_ratio}, "
            f"true_pert={args.true_perturbation_fraction:.3f}, "
            f"t2f={args.false_perturb_t_to_f_ratio}, "
            f"eq_methods={args.equal_methods}, "
            f"method_mode={args.method_balance_mode})"
        ),
    )

    if args.output_json is not None:
        write_output_json(args.output_json, result)
        print(f"\n  Wrote sampled dataset JSON to: {args.output_json}")

    # ------- Stage 5: multi-config comparison -------
    if args.compare:
        import copy

        report_comparison(
            [copy.deepcopy(c) for c in PRESET_CONFIGS],
            json_files,
        )

    print(f"\n{SEP}")
    print("  DONE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
