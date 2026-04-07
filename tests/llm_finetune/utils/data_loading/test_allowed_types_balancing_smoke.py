"""Smoke test for selective perturbation loading and method balancing."""

from __future__ import annotations

import json
from pathlib import Path

from src.llm_finetune.utils.data_loading.perturbation_types import (
    parse_file_config,
)
from tests.sanity_checks.check_allowed_types_balancing import run_validation


def _make_record(
    claim_id: str,
    label: str,
    normalized_claim: str,
    method: str,
    perturb_fields: dict[str, str],
) -> dict:
    return {
        "claim_id": claim_id,
        "label": label,
        "normalized_claim": normalized_claim,
        "perturbation_method": method,
        "claim": normalized_claim,
        "doc": "Evidence text.",
        **perturb_fields,
    }


def _write_json(data: list[dict], directory: Path, name: str) -> Path:
    path = directory / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return path


def test_selective_loading_and_method_balancing_smoke(tmp_path):
    approximation_true = _write_json(
        [
            _make_record(
                "approx_t_1",
                "True",
                "Approximation true claim 1.",
                "approximation",
                {
                    "perturb_type1_True": "Approximation type1 true 1.",
                    "perturb_type2_True": "Approximation type2 true 1.",
                },
            ),
            _make_record(
                "approx_t_2",
                "True",
                "Approximation true claim 2.",
                "approximation",
                {
                    "perturb_type1_True": "Approximation type1 true 2.",
                    "perturb_type2_True": "Approximation type2 true 2.",
                },
            ),
        ],
        tmp_path,
        "approximation_True.json",
    )
    approximation_false = _write_json(
        [
            _make_record(
                "approx_f_1",
                "False",
                "Approximation false claim 1.",
                "approximation",
                {
                    "perturb_type1_False": "Approximation type1 false 1.",
                    "perturb_type2_False": "Approximation type2 false 1.",
                },
            ),
        ],
        tmp_path,
        "approximation_False.json",
    )
    range_true = _write_json(
        [
            _make_record(
                "range_t_1",
                "True",
                "Range true claim 1.",
                "range",
                {
                    "perturb_type1_True": "Range type1 true 1.",
                    "perturb_type2_True": "Range type2 true 1.",
                },
            ),
        ],
        tmp_path,
        "range_True.json",
    )
    range_false = _write_json(
        [
            _make_record(
                "range_f_1",
                "False",
                "Range false claim 1.",
                "range",
                {
                    "perturb_type1_False": "Range type1 false 1.",
                    "perturb_type2_False": "Range type2 false 1.",
                },
            ),
            _make_record(
                "range_f_2",
                "False",
                "Range false claim 2.",
                "range",
                {
                    "perturb_type1_False": "Range type1 false 2.",
                    "perturb_type2_False": "Range type2 false 2.",
                },
            ),
        ],
        tmp_path,
        "range_False.json",
    )
    negative_true = _write_json(
        [
            _make_record(
                "negative_t_1",
                "True",
                "Negative true claim 1.",
                "negative",
                {
                    "perturb_type2_True": "Negative type2 true 1.",
                },
            ),
        ],
        tmp_path,
        "negative_True.json",
    )
    negative_false = _write_json(
        [
            _make_record(
                "negative_f_1",
                "False",
                "Negative false claim 1.",
                "negative",
                {
                    "perturb_type2_False": "Negative type2 false 1.",
                },
            ),
            _make_record(
                "negative_f_2",
                "False",
                "Negative false claim 2.",
                "negative",
                {
                    "perturb_type2_False": "Negative type2 false 2.",
                },
            ),
        ],
        tmp_path,
        "negative_False.json",
    )
    obfuscation_true = _write_json(
        [
            _make_record(
                "random_t_1",
                "True",
                "Random true claim 1.",
                "random",
                {
                    "perturb_type3_random_True": "Random perturb true 1.",
                },
            ),
            _make_record(
                "random_t_2",
                "True",
                "Random true claim 2.",
                "random",
                {
                    "perturb_type3_random_True": "Random perturb true 2.",
                },
            ),
            _make_record(
                "masked_t_1",
                "True",
                "Masked true claim 1.",
                "masked",
                {
                    "perturb_type3_masked_True": "Masked perturb true 1.",
                },
            ),
        ],
        tmp_path,
        "obfuscation_True.json",
    )
    obfuscation_false = _write_json(
        [
            _make_record(
                "random_f_1",
                "False",
                "Random false claim 1.",
                "random",
                {
                    "perturb_type3_random_False": "Random perturb false 1.",
                },
            ),
            _make_record(
                "masked_f_1",
                "False",
                "Masked false claim 1.",
                "masked",
                {
                    "perturb_type3_masked_False": "Masked perturb false 1.",
                },
            ),
            _make_record(
                "masked_f_2",
                "False",
                "Masked false claim 2.",
                "masked",
                {
                    "perturb_type3_masked_False": "Masked perturb false 2.",
                },
            ),
            _make_record(
                "masked_f_3",
                "False",
                "Masked false claim 3.",
                "masked",
                {
                    "perturb_type3_masked_False": "Masked perturb false 3.",
                },
            ),
        ],
        tmp_path,
        "obfuscation_False.json",
    )

    config_strings = [
        f"{approximation_true}:type1,type2",
        f"{approximation_false}:type1,type2",
        f"{range_true}:type1,type2",
        f"{range_false}:type1",
        f"{negative_true}:type2",
        f"{negative_false}:type2",
        f"{obfuscation_true}:type3_random,type3_masked",
        f"{obfuscation_false}:type3_random,type3_masked",
    ]
    file_configs = [parse_file_config(config) for config in config_strings]

    validation = run_validation(
        file_configs=file_configs,
        method_balance_modes=["none", "oversample", "undersample"],
        balance_labels=False,
        perturb_ratio=100.0,
        false_perturb_t_to_f_ratio=None,
        seed=42,
    )

    assert validation["all_failures"] == []
    assert validation["filter_failures"] == []

    raw_counts = validation["raw_counts"]
    filtered_counts = validation["filtered_counts"]
    assert raw_counts["approximation_False.json"]["type2"] == 1
    assert filtered_counts["approximation_False.json"]["type2"] == 1
    assert raw_counts["range_False.json"]["type2"] == 2
    assert filtered_counts["range_False.json"].get("type2", 0) == 0

    baseline_counts = {
        sample_type: dict(method_counts)
        for sample_type, method_counts in validation["baseline_counts"].items()
    }
    assert baseline_counts == {
        "T→T": {"approximation": 2, "range": 1},
        "T→F": {
            "approximation": 2,
            "range": 1,
            "negative": 1,
            "random": 2,
            "masked": 1,
        },
        "F→F": {
            "approximation": 2,
            "range": 2,
            "negative": 2,
            "random": 1,
            "masked": 3,
        },
    }

    oversample_counts = {
        sample_type: dict(method_counts)
        for sample_type, method_counts in validation["mode_results"][
            "oversample"
        ]["counts"].items()
    }
    assert oversample_counts == {
        "T→T": {"approximation": 2, "range": 2},
        "T→F": {
            "approximation": 2,
            "range": 2,
            "negative": 2,
            "random": 2,
            "masked": 2,
        },
        "F→F": {
            "approximation": 3,
            "range": 3,
            "negative": 3,
            "random": 3,
            "masked": 3,
        },
    }

    undersample_counts = {
        sample_type: dict(method_counts)
        for sample_type, method_counts in validation["mode_results"][
            "undersample"
        ]["counts"].items()
    }
    assert undersample_counts == {
        "T→T": {"approximation": 1, "range": 1},
        "T→F": {
            "approximation": 1,
            "range": 1,
            "negative": 1,
            "random": 1,
            "masked": 1,
        },
        "F→F": {
            "approximation": 1,
            "range": 1,
            "negative": 1,
            "random": 1,
            "masked": 1,
        },
    }
