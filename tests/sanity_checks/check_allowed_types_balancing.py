"""Sanity check selective perturbation loading and method balancing.

Defaults to the train split configuration that excludes ``type2`` from
approximation/range False files while keeping the other requested types.

The script validates two things:

1. Only allowed perturbation types survive per-file filtering.
2. Method balancing behaves as expected for ``none``, ``oversample``, and
   ``undersample`` within each perturbation class (``T→T``, ``T→F``, ``F→F``).

Usage::

    python -m tests.sanity_checks.check_allowed_types_balancing

    python -m tests.sanity_checks.check_allowed_types_balancing \
        --method_balance_modes oversample undersample

    python -m tests.sanity_checks.check_allowed_types_balancing \
        --file-config \
            ./data/perturbed_data/train_split/approximation_True.json:type1,type2 \
            ./data/perturbed_data/train_split/approximation_False.json:type1,type2
"""  # noqa: E501

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, cast

from src.llm_finetune.utils.data_loading.dataset_balancer import (
    Sample,
    build_balanced_dataset,
    deduplicate,
    flatten_samples,
    load_files,
)
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
    parse_file_config,
)

SEP = "=" * 80
THIN_SEP = "-" * 80

DEFAULT_FILE_CONFIGS = [
    "./data/perturbed_data/train_split/approximation_True.json:type1,type2",
    "./data/perturbed_data/train_split/approximation_False.json:type1",
    "./data/perturbed_data/train_split/range_True.json:type1,type2",
    "./data/perturbed_data/train_split/range_False.json:type1",
    "./data/perturbed_data/train_split/negative_True.json:type2",
    "./data/perturbed_data/train_split/negative_False.json:type2",
    (
        "./data/perturbed_data/train_split/obfuscation_True.json:"
        "type3_random,type3_masked"
    ),
    (
        "./data/perturbed_data/train_split/obfuscation_False.json:"
        "type3_random,type3_masked"
    ),
]


def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _subheader(title: str) -> None:
    print(f"\n  {THIN_SEP}")
    print(f"  {title}")
    print(f"  {THIN_SEP}")


def _build_allowed_types_by_file(
    file_configs: list[FileLoadConfig],
) -> dict[str, set[str]] | None:
    """Mirror pipeline behaviour for filename-based allowed type filters."""
    if not file_configs:
        return None

    if not any(cfg.perturb_types is not None for cfg in file_configs):
        return None

    mapping: dict[str, set[str]] = {}
    for cfg in file_configs:
        if cfg.perturb_types is None:
            continue
        mapping[Path(cfg.path).name] = {
            perturb_type.value.removeprefix("perturb_")
            for perturb_type in cfg.perturb_types
            if perturb_type != PerturbationType.ORIGINAL_NORMALIZED
        }
    return mapping


def _parse_configs(config_strings: list[str]) -> list[FileLoadConfig]:
    return [parse_file_config(config) for config in config_strings]


def _validate_paths_exist(file_paths: list[str]) -> None:
    missing = [path for path in file_paths if not Path(path).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing input files:\n{missing_text}")


def _perturbation_counts_by_file(
    samples: list[Sample],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if not sample.is_perturbed or sample.perturb_type_key is None:
            continue
        counts[sample.source_file][sample.perturb_type_key] += 1
    return counts


def _method_counts_by_type_from_samples(
    samples: list[Sample],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if not sample.is_perturbed:
            continue
        method = sample.perturbation_method or "(missing)"
        counts[sample.type][method] += 1
    return counts


def _method_counts_by_type_from_dicts(
    samples: list[dict],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if not sample["is_perturbed"]:
            continue
        method = sample["perturbation_method"] or "(missing)"
        counts[sample["type"]][method] += 1
    return counts


def _print_file_type_counts(
    title: str,
    counts: dict[str, Counter[str]],
    allowed_types_by_file: dict[str, set[str]] | None,
) -> None:
    _subheader(title)
    print(f"  {'file':<30s}  {'type':<18s}  {'count':>7s}  {'status':<10s}")
    print(f"  {THIN_SEP}")

    for source_file in sorted(counts):
        allowed = (
            None
            if allowed_types_by_file is None
            else allowed_types_by_file.get(source_file)
        )
        for perturb_type, count in sorted(counts[source_file].items()):
            status = "allowed"
            if allowed is not None and perturb_type not in allowed:
                status = "blocked"
            print(
                f"  {source_file:<30s}  {perturb_type:<18s}  "
                f"{count:>7d}  {status:<10s}"
            )


def _print_method_type_counts(
    title: str,
    counts: dict[str, Counter[str]],
) -> None:
    _subheader(title)
    all_methods = sorted(
        {
            method
            for method_counts in counts.values()
            for method in method_counts
        }
    )
    all_types = [
        sample_type
        for sample_type in ("T→T", "T→F", "F→F")
        if sample_type in counts
    ]

    if not all_methods:
        print("  No perturbed samples found.")
        return

    header = f"  {'method':<18s}"
    for sample_type in all_types:
        header += f"  {sample_type:>8s}"
    header += f"  {'TOTAL':>8s}"
    print(header)
    print(f"  {THIN_SEP}")

    for method in all_methods:
        row_total = 0
        row = f"  {method:<18s}"
        for sample_type in all_types:
            count = counts[sample_type].get(method, 0)
            row_total += count
            row += f"  {count:>8d}"
        row += f"  {row_total:>8d}"
        print(row)


def _expected_counts_for_mode(
    baseline_counts: dict[str, Counter[str]],
    mode: str,
) -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {}
    for sample_type, method_counts in baseline_counts.items():
        if not method_counts:
            continue
        if mode == "none":
            expected[sample_type] = dict(method_counts)
            continue

        reducer = max if mode == "oversample" else min
        target = reducer(method_counts.values())
        expected[sample_type] = {method: target for method in method_counts}
    return expected


def _validate_filtering(
    raw_counts: dict[str, Counter[str]],
    filtered_counts: dict[str, Counter[str]],
    allowed_types_by_file: dict[str, set[str]] | None,
) -> list[str]:
    failures: list[str] = []
    if allowed_types_by_file is None:
        return failures

    for source_file, allowed_types in sorted(allowed_types_by_file.items()):
        raw_for_file = raw_counts.get(source_file, Counter())
        filtered_for_file = filtered_counts.get(source_file, Counter())

        for perturb_type, filtered_count in filtered_for_file.items():
            if perturb_type not in allowed_types and filtered_count > 0:
                failures.append(
                    f"{source_file}: disallowed {perturb_type} survived "
                    f"filtering ({filtered_count} samples)"
                )

        for perturb_type, raw_count in raw_for_file.items():
            filtered_count = filtered_for_file.get(perturb_type, 0)
            if (
                perturb_type in allowed_types
                and raw_count > 0
                and filtered_count == 0
            ):
                failures.append(
                    f"{source_file}: allowed {perturb_type} disappeared "
                    "completely after filtering"
                )
    return failures


def _validate_method_balancing(
    mode: str,
    baseline_counts: dict[str, Counter[str]],
    actual_counts: dict[str, Counter[str]],
) -> list[str]:
    failures: list[str] = []
    expected = _expected_counts_for_mode(baseline_counts, mode)

    for sample_type, expected_counts in expected.items():
        actual_for_type = dict(actual_counts.get(sample_type, Counter()))
        if actual_for_type != expected_counts:
            failures.append(
                f"{mode} {sample_type}: expected {expected_counts}, "
                f"got {actual_for_type}"
            )

    extra_types = set(actual_counts) - set(expected)
    for sample_type in sorted(extra_types):
        failures.append(
            f"{mode} {sample_type}: unexpected sample type present "
            f"with counts {dict(actual_counts[sample_type])}"
        )

    return failures


def _print_validation_result(title: str, failures: list[str]) -> None:
    _subheader(title)
    if not failures:
        print("  PASS")
        return

    print("  FAIL")
    for failure in failures:
        print(f"    - {failure}")


def run_validation(
    file_configs: list[FileLoadConfig],
    method_balance_modes: list[str] | None = None,
    balance_labels: bool = False,
    perturb_ratio: float = 100.0,
    false_perturb_t_to_f_ratio: float | None = None,
    true_perturbation_fraction: float = 1 / 3,
    seed: int = 42,
) -> dict:
    """Run selective-loading and method-balancing validation.

    Returns a dictionary with the computed counts and any validation failures so
    callers can either print a report or assert in tests.
    """
    resolved_modes = method_balance_modes or [
        "none",
        "oversample",
        "undersample",
    ]
    file_paths = [config.path for config in file_configs]
    _validate_paths_exist(file_paths)

    allowed_types_by_file = _build_allowed_types_by_file(file_configs)
    file_data = load_files(cast(list[str | Path], file_paths))

    raw_flat = flatten_samples(file_data)
    filtered_flat = flatten_samples(
        file_data,
        allowed_types_by_file=allowed_types_by_file,
    )
    filtered_deduped = deduplicate(filtered_flat)

    raw_counts = _perturbation_counts_by_file(raw_flat)
    filtered_counts = _perturbation_counts_by_file(filtered_flat)
    baseline_counts = _method_counts_by_type_from_samples(filtered_deduped)
    filter_failures = _validate_filtering(
        raw_counts,
        filtered_counts,
        allowed_types_by_file,
    )

    mode_results: dict[str, dict] = {}
    all_failures = list(filter_failures)

    for mode in resolved_modes:
        result = build_balanced_dataset(
            file_paths=cast(list[str | Path], file_paths),
            balance_labels=balance_labels,
            perturb_ratio=perturb_ratio,
            false_perturb_t_to_f_ratio=false_perturb_t_to_f_ratio,
            true_perturbation_fraction=true_perturbation_fraction,
            method_balance_mode=cast(
                Literal["none", "undersample", "oversample"], mode
            ),
            seed=seed,
            allowed_types_by_file=allowed_types_by_file,
        )
        counts = _method_counts_by_type_from_dicts(result)
        failures = _validate_method_balancing(
            mode,
            baseline_counts,
            counts,
        )
        mode_results[mode] = {
            "samples": result,
            "counts": counts,
            "failures": failures,
        }
        all_failures.extend(failures)

    return {
        "file_configs": file_configs,
        "allowed_types_by_file": allowed_types_by_file,
        "raw_counts": raw_counts,
        "filtered_counts": filtered_counts,
        "baseline_counts": baseline_counts,
        "filter_failures": filter_failures,
        "mode_results": mode_results,
        "all_failures": all_failures,
    }


def main() -> None:
    """Entry point for CLI usage.

    Parses arguments, runs validation, and prints a report. Exits with code 1 if
    any validation failures are found.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check selective type loading and method balancing."
        )
    )
    parser.add_argument(
        "--file-config",
        nargs="+",
        default=None,
        help=(
            "CLI-style file configs, e.g. path.json:type1,type2. "
            "Defaults to the train configuration used for selective loading."
        ),
    )
    parser.add_argument(
        "--method_balance_modes",
        nargs="+",
        choices=["none", "oversample", "undersample"],
        default=["none", "oversample", "undersample"],
        help="Which method balance modes to validate.",
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        default=False,
        help=(
            "Also enable label balancing. Off by default to isolate "
            "method balancing."
        ),
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=100.0,
        help=(
            "Perturbed/original ratio passed to build_balanced_dataset. "
            "The default is large so only method balancing changes the output."
        ),
    )
    parser.add_argument(
        "--false_perturb_t_to_f_ratio",
        type=float,
        default=None,
        help=(
            "Optional False-perturbation composition constraint. "
            "Default: disabled so only method balancing is tested."
        ),
    )
    parser.add_argument(
        "--true_perturbation_fraction",
        type=float,
        default=1 / 3,
        help="Share of perturbed examples that should remain True.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    config_strings = args.file_config or DEFAULT_FILE_CONFIGS
    file_configs = _parse_configs(config_strings)
    validation = run_validation(
        file_configs=file_configs,
        method_balance_modes=args.method_balance_modes,
        balance_labels=args.balance_labels,
        perturb_ratio=args.perturb_ratio,
        false_perturb_t_to_f_ratio=args.false_perturb_t_to_f_ratio,
        true_perturbation_fraction=args.true_perturbation_fraction,
        seed=args.seed,
    )
    allowed_types_by_file = validation["allowed_types_by_file"]
    raw_counts = validation["raw_counts"]
    filtered_counts = validation["filtered_counts"]
    baseline_counts = validation["baseline_counts"]
    filter_failures = validation["filter_failures"]
    mode_results = validation["mode_results"]
    all_failures = validation["all_failures"]

    _header("ALLOWED TYPES AND METHOD BALANCING SANITY CHECK")
    print("  File configs:")
    for config in file_configs:
        print(f"    {config}")

    _subheader("Allowed types by file")
    if allowed_types_by_file is None:
        print("  No per-file type restrictions configured.")
    else:
        for source_file, allowed_types in sorted(allowed_types_by_file.items()):
            joined = ", ".join(sorted(allowed_types))
            print(f"  {source_file:<30s}  ->  {joined}")

    _print_file_type_counts(
        "Raw parsed perturbation counts",
        raw_counts,
        allowed_types_by_file,
    )
    _print_file_type_counts(
        "Filtered perturbation counts",
        filtered_counts,
        allowed_types_by_file,
    )

    filter_failures = _validate_filtering(
        raw_counts,
        filtered_counts,
        allowed_types_by_file,
    )
    _print_validation_result("Allowed-type filtering", filter_failures)

    _print_method_type_counts(
        "Baseline deduplicated method counts by perturbation class",
        baseline_counts,
    )

    for mode in args.method_balance_modes:
        counts = mode_results[mode]["counts"]
        _print_method_type_counts(
            f"Balanced output for method_balance_mode={mode}",
            counts,
        )

        mode_failures = mode_results[mode]["failures"]
        _print_validation_result(
            f"Method balancing: {mode}",
            mode_failures,
        )

    _header("SUMMARY")
    if all_failures:
        print(f"  FAILED with {len(all_failures)} issue(s).")
        raise SystemExit(1)

    print("  PASS")


if __name__ == "__main__":
    main()
