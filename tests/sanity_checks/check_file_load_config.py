"""Sanity check: FileLoadConfig + expand_perturbations filtering.

Run this script to inspect exactly which perturbation types and labels are
included or excluded for a given set of FileLoadConfig entries.  Useful for
verifying shell-script :type1,type2 filter strings before launching training.

Usage::

    # From project root (with venv active):
    python -m tests.sanity_checks.check_file_load_config

    # Or with custom configs on the CLI:
    python -m tests.sanity_checks.check_file_load_config \
        "data/perturbed_data/train_split/numeration_True.json:type1,type2" \
        "data/perturbed_data/train_split/numeration_False.json:type1"
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List

from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
    parse_file_config,
)
from src.llm_finetune.utils.data_loading.perturbation_logic import (
    _PERTURB_KEY_RE,
    expand_perturbations,
)
from src.llm_finetune.utils.data_loading.loader import (
    load_json_files,
)

# ── Default configs to check (mirrors the enhanced-batch training config) ──
DEFAULT_CONFIGS: List[str] = [
    "data/perturbed_data/train_split/numeration_True.json:type1,type2",
    "data/perturbed_data/train_split/numeration_False.json:type1",
]

SEPARATOR = "=" * 72


def _available_perturb_keys(examples: list[dict]) -> list[str]:
    """Return sorted unique perturbation keys present across all examples."""
    keys: set[str] = set()
    for ex in examples:
        for k in ex:
            if _PERTURB_KEY_RE.match(k):
                keys.add(k)
    return sorted(keys)


def check_config(config: FileLoadConfig) -> None:
    """Load data for a single FileLoadConfig and print what is included."""
    path = Path(config.path)
    print(SEPARATOR)
    print(f"FILE: {path}")
    print(SEPARATOR)

    if not path.exists():
        print("  ✗ File not found — skipping.\n")
        return

    raw = load_json_files(str(path))
    print(f"  Raw examples loaded: {len(raw)}")

    # Show all perturbation keys found in the file
    available_keys = _available_perturb_keys(raw)
    print(f"  Perturbation keys in file: {available_keys}")

    # Show what the filter is
    if config.perturb_types is None:
        filter_label = "ALL (no filter)"
    else:
        filter_label = ", ".join(t.value for t in config.perturb_types)
    print(f"  Filter applied: {filter_label}")

    # Expand and count
    expanded = expand_perturbations(raw, allowed_types=config.perturb_types)
    print(f"  Expanded examples: {len(expanded)}")

    # per-type counts
    type_counts = Counter(ex["perturbation_type"] for ex in expanded)
    label_counts = Counter(ex["label"] for ex in expanded)

    print()
    print("  INCLUDED perturbation types:")
    for ptype, count in sorted(type_counts.items()):
        print(f"    {ptype:<30s}  {count:>5d} examples")

    print()
    print("  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:<10s}  {count:>5d}")

    # Show what was excluded
    all_types = {PerturbationType(k.rsplit("_", 1)[0]) for k in available_keys}
    if config.perturb_types is not None:
        included = set(config.perturb_types)
        excluded = all_types - included
    else:
        excluded = set()

    print()
    if excluded:
        print("  EXCLUDED perturbation types:")
        for t in sorted(excluded, key=lambda x: x.value):
            print(f"    ✗ {t.value}")
    else:
        print("  EXCLUDED: (none — all types loaded)")

    print()


def main(config_strings: List[str] | None = None) -> None:  # noqa: C901
    """Run the sanity check for a list of config strings."""
    if config_strings is None:
        config_strings = DEFAULT_CONFIGS

    configs = [parse_file_config(s) for s in config_strings]

    print()
    print(SEPARATOR)
    print("  FileLoadConfig sanity check")
    print(SEPARATOR)
    print()
    print(f"  Configs to check: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"    [{i}] {cfg}")
    print()

    for cfg in configs:
        check_config(cfg)

    # Summary across all configs
    print(SEPARATOR)
    print("  COMBINED SUMMARY")
    print(SEPARATOR)

    all_expanded: list[dict] = []
    for cfg in configs:
        path = Path(cfg.path)
        if not path.exists():
            continue
        raw = load_json_files(str(path))
        expanded = expand_perturbations(raw, allowed_types=cfg.perturb_types)
        for ex in expanded:
            ex["source_file"] = path.name
        all_expanded.extend(expanded)

    total_type_counts = Counter(ex["perturbation_type"] for ex in all_expanded)
    total_label_counts = Counter(ex["label"] for ex in all_expanded)

    print(f"  Total expanded examples: {len(all_expanded)}")
    print()
    print("  By perturbation type:")
    for ptype, count in sorted(total_type_counts.items()):
        print(f"    {ptype:<30s}  {count:>5d}")
    print()
    print("  By label:")
    for label, count in sorted(total_label_counts.items()):
        print(f"    {label:<10s}  {count:>5d}")
    print()

    # Cross-tab: source_file × perturbation_type
    print("  By source file × perturbation type:")
    file_type_counts = Counter(
        (ex["source_file"], ex["perturbation_type"]) for ex in all_expanded
    )
    current_file = None
    for (sf, pt), count in sorted(file_type_counts.items()):
        if sf != current_file:
            print(f"    {sf}:")
            current_file = sf
        print(f"      {pt:<30s}  {count:>5d}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)
