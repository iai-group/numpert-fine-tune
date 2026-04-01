"""Tests for the dataset balancer module."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pytest

from src.llm_finetune.utils.data_loading.balancer.dataset_balancer import (
    Sample,
    _assign_type,
    _compute_true_false_perturbation_targets,
    _determine_perturbed_label,
    _oversample_fair,
    _parse_perturb_key,
    _undersample_preserving_method_balance,
    _undersample_fair,
    apply_balancing,
    build_balanced_dataset,
    compute_stats,
    deduplicate,
    enforce_coupling,
    flatten_samples,
    load_files,
)

# ── Helpers ──────────────────────────────────────────────────────


def _make_record(
    claim_id: str,
    label: str,
    normalized_claim: str,
    method: str = "numeration",
    perturb_fields: Optional[dict[str, str]] = None,
) -> dict:
    """Create a minimal input record."""
    rec = {
        "claim_id": claim_id,
        "label": label,
        "normalized_claim": normalized_claim,
        "pertubation_method": method,
        "claim": normalized_claim,
        "doc": "Evidence text.",
    }
    if perturb_fields:
        rec.update(perturb_fields)
    return rec


def _write_json(data: list[dict], directory: Path, name: str) -> Path:
    """Write data to a JSON file and return the path."""
    path = directory / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _make_sample(
    text: str = "claim text",
    label: str = "True",
    claim_id: str = "c_001",
    sample_type: str = "Original",
    method: Optional[str] = None,
    is_perturbed: bool = False,
    source_file: str = "file_a.json",
) -> Sample:
    return Sample(
        text=text,
        label=label,
        source_claim_id=claim_id,
        type=sample_type,
        perturbation_method=method,
        is_perturbed=is_perturbed,
        source_file=source_file,
    )


# ── _parse_perturb_key ──────────────────────────────────────────


class TestParseKey:
    def test_type1_true(self):
        assert _parse_perturb_key("perturb_type1_True") == (
            "type1",
            "True",
        )

    def test_type2_false(self):
        assert _parse_perturb_key("perturb_type2_False") == (
            "type2",
            "False",
        )

    def test_type3_random_true(self):
        assert _parse_perturb_key("perturb_type3_random_True") == (
            "type3_random",
            "True",
        )

    def test_type3_masked_false(self):
        assert _parse_perturb_key("perturb_type3_masked_False") == (
            "type3_masked",
            "False",
        )

    def test_type3_removed_true(self):
        assert _parse_perturb_key("perturb_type3_removed_True") == (
            "type3_removed",
            "True",
        )

    def test_non_perturb_key_returns_none(self):
        assert _parse_perturb_key("pertubation_method") is None
        assert _parse_perturb_key("claim_id") is None
        assert _parse_perturb_key("label") is None


# ── _determine_perturbed_label ──────────────────────────────────


class TestDetermineLabel:
    def test_type1_preserves_true(self):
        assert _determine_perturbed_label("True", "type1") == "True"

    def test_type1_preserves_false(self):
        assert _determine_perturbed_label("False", "type1") == "False"

    def test_type2_always_false(self):
        assert _determine_perturbed_label("True", "type2") == "False"
        assert _determine_perturbed_label("False", "type2") == "False"

    def test_type3_always_false(self):
        assert _determine_perturbed_label("True", "type3_random") == "False"
        assert _determine_perturbed_label("False", "type3_masked") == "False"
        assert _determine_perturbed_label("True", "type3_removed") == "False"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation"):
            _determine_perturbed_label("True", "type99")


# ── _assign_type ────────────────────────────────────────────────


class TestAssignType:
    def test_t_to_t(self):
        assert _assign_type("True", "True") == "T→T"

    def test_t_to_f(self):
        assert _assign_type("True", "False") == "T→F"

    def test_f_to_f(self):
        assert _assign_type("False", "False") == "F→F"

    def test_f_to_t_raises(self):
        with pytest.raises(ValueError, match="Unexpected"):
            _assign_type("False", "True")


# ── load_files ──────────────────────────────────────────────────


class TestLoadFiles:
    def test_loads_multiple_files(self, tmp_path):
        data_a = [_make_record("c1", "True", "claim one")]
        data_b = [_make_record("c2", "False", "claim two")]
        path_a = _write_json(data_a, tmp_path, "a.json")
        path_b = _write_json(data_b, tmp_path, "b.json")

        result = load_files([path_a, path_b])
        assert len(result) == 2
        assert result[0][0] == "a.json"
        assert result[1][0] == "b.json"
        assert len(result[0][1]) == 1
        assert len(result[1][1]) == 1

    def test_empty_file(self, tmp_path):
        path = _write_json([], tmp_path, "empty.json")
        result = load_files([path])
        assert result == [("empty.json", [])]


# ── flatten_samples ─────────────────────────────────────────────


class TestFlattenSamples:
    def test_original_only(self):
        records = [_make_record("c1", "True", "The number is 42.")]
        samples = flatten_samples([("file.json", records)])
        assert len(samples) == 1
        s = samples[0]
        assert s.text == "The number is 42."
        assert s.label == "True"
        assert s.type == "Original"
        assert s.is_perturbed is False
        assert s.source_file == "file.json"

    def test_with_perturbations(self):
        records = [
            _make_record(
                "c1",
                "True",
                "The number is 42.",
                method="numeration",
                perturb_fields={
                    "perturb_type1_True": "The number is forty-two.",
                    "perturb_type2_True": "The number is forty-five.",
                },
            )
        ]
        samples = flatten_samples([("file.json", records)])
        assert len(samples) == 3  # 1 original + 2 perturbed

        originals = [s for s in samples if not s.is_perturbed]
        perturbed = [s for s in samples if s.is_perturbed]
        assert len(originals) == 1
        assert len(perturbed) == 2

        # type1 preserves label -> T→T
        t1 = [s for s in perturbed if "forty-two" in s.text][0]
        assert t1.label == "True"
        assert t1.type == "T→T"

        # type2 always False -> T→F
        t2 = [s for s in perturbed if "forty-five" in s.text][0]
        assert t2.label == "False"
        assert t2.type == "T→F"

    def test_false_label_perturbs(self):
        records = [
            _make_record(
                "c2",
                "False",
                "Claim is false.",
                method="approximation",
                perturb_fields={
                    "perturb_type1_False": "type1 perturbed text",
                    "perturb_type2_False": "type2 perturbed text",
                },
            )
        ]
        samples = flatten_samples([("f.json", records)])
        perturbed = [s for s in samples if s.is_perturbed]
        assert len(perturbed) == 2

        t1 = [s for s in perturbed if "type1" in s.text][0]
        assert t1.label == "False"  # type1 preserves label
        assert t1.type == "F→F"

        t2 = [s for s in perturbed if "type2" in s.text][0]
        assert t2.label == "False"  # type2 -> False
        assert t2.type == "F→F"

    def test_type3_obfuscation_variants(self):
        records = [
            _make_record(
                "c3",
                "True",
                "Claim with numbers.",
                method="random",
                perturb_fields={
                    "perturb_type3_random_True": "random perturbed",
                    "perturb_type3_masked_True": "masked perturbed",
                    "perturb_type3_removed_True": "removed perturbed",
                },
            )
        ]
        samples = flatten_samples([("f.json", records)])
        perturbed = [s for s in samples if s.is_perturbed]
        # "removed" is excluded, so only random + masked
        assert len(perturbed) == 2
        texts = {s.text for s in perturbed}
        assert "random perturbed" in texts
        assert "masked perturbed" in texts
        assert "removed perturbed" not in texts
        for s in perturbed:
            assert s.label == "False"
            assert s.type == "T→F"

    def test_label_mismatch_skips(self):
        """perturb_type1_False on a True-label record is skipped."""
        records = [
            _make_record(
                "c4",
                "True",
                "Claim text.",
                perturb_fields={"perturb_type1_False": "mismatched perturb"},
            )
        ]
        samples = flatten_samples([("f.json", records)])
        assert len(samples) == 1  # Only original
        assert samples[0].type == "Original"

    def test_empty_perturb_value_skipped(self):
        records = [
            _make_record(
                "c5",
                "True",
                "Claim.",
                perturb_fields={
                    "perturb_type1_True": "",
                    "perturb_type2_True": "   ",
                },
            )
        ]
        samples = flatten_samples([("f.json", records)])
        assert len(samples) == 1  # Only original

    def test_missing_required_fields_skipped(self):
        records = [
            {"claim_id": "c6"},  # missing label and claim
        ]
        samples = flatten_samples([("f.json", records)])
        assert len(samples) == 0

    def test_perturbation_method_tracked(self):
        records = [
            _make_record(
                "c7",
                "True",
                "Claim.",
                method="approximation",
                perturb_fields={
                    "perturb_type1_True": "perturbed",
                },
            )
        ]
        samples = flatten_samples([("f.json", records)])
        perturbed = [s for s in samples if s.is_perturbed]
        assert perturbed[0].perturbation_method == "approximation"


# ── deduplicate ─────────────────────────────────────────────────


class TestDeduplicate:
    def test_removes_exact_text_duplicates(self):
        samples = [
            _make_sample(text="A", claim_id="c1"),
            _make_sample(text="B", claim_id="c2"),
            _make_sample(text="A", claim_id="c3"),  # duplicate
        ]
        result = deduplicate(samples)
        assert len(result) == 2
        assert {s.text for s in result} == {"A", "B"}

    def test_keeps_first_occurrence(self):
        s1 = _make_sample(text="Same", claim_id="first", source_file="f1.json")
        s2 = _make_sample(text="Same", claim_id="second", source_file="f2.json")
        result = deduplicate([s1, s2])
        assert len(result) == 1
        assert result[0].source_claim_id == "first"

    def test_no_duplicates_passthrough(self):
        samples = [
            _make_sample(text="A"),
            _make_sample(text="B"),
            _make_sample(text="C"),
        ]
        result = deduplicate(samples)
        assert len(result) == 3


# ── _undersample_fair ───────────────────────────────────────────


class TestUndersampleFair:
    def test_no_reduction_needed(self):
        import random

        rng = random.Random(42)
        items = [_make_sample(text=f"t{i}") for i in range(3)]
        result = _undersample_fair(items, 5, rng)
        assert len(result) == 3

    def test_reduces_to_target(self):
        import random

        rng = random.Random(42)
        items = [
            _make_sample(text=f"t{i}", source_file=f"file_{i % 3}.json")
            for i in range(10)
        ]
        result = _undersample_fair(items, 5, rng)
        assert len(result) == 5

    def test_multi_file_fairness(self):
        import random

        rng = random.Random(42)
        items = []
        # 10 from file_a, 10 from file_b
        for i in range(10):
            items.append(_make_sample(text=f"a_{i}", source_file="file_a.json"))
            items.append(_make_sample(text=f"b_{i}", source_file="file_b.json"))

        result = _undersample_fair(items, 6, rng)
        file_counts = Counter(s.source_file for s in result)
        assert file_counts["file_a.json"] == 3
        assert file_counts["file_b.json"] == 3


class TestOversampleFair:
    def test_no_expansion_needed(self):
        import random

        rng = random.Random(42)
        items = [_make_sample(text=f"t{i}") for i in range(3)]
        result = _oversample_fair(items, 2, rng)
        assert len(result) == 3

    def test_expands_to_target(self):
        import random

        rng = random.Random(42)
        items = [
            _make_sample(text=f"t{i}", source_file=f"file_{i % 2}.json")
            for i in range(3)
        ]
        result = _oversample_fair(items, 7, rng)
        assert len(result) == 7
        assert {s.text for s in items} <= {s.text for s in result}

    def test_multi_file_fairness(self):
        import random

        rng = random.Random(42)
        items = [
            _make_sample(text="a_0", source_file="file_a.json"),
            _make_sample(text="b_0", source_file="file_b.json"),
        ]

        result = _oversample_fair(items, 6, rng)
        file_counts = Counter(s.source_file for s in result)
        assert file_counts["file_a.json"] == 3
        assert file_counts["file_b.json"] == 3


class TestUndersamplePreservingMethodBalance:
    def test_balances_methods_as_evenly_as_possible(self):
        import random

        rng = random.Random(42)
        items = []
        for i in range(5):
            items.append(
                _make_sample(
                    text=f"range_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            items.append(
                _make_sample(
                    text=f"numeration_{i}",
                    label="False",
                    sample_type="T→F",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            items.append(
                _make_sample(
                    text=f"approx_{i}",
                    label="False",
                    sample_type="T→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )

        result = _undersample_preserving_method_balance(items, 8, rng)
        counts = Counter(s.perturbation_method for s in result)

        assert len(result) == 8
        assert max(counts.values()) - min(counts.values()) <= 1


class TestComputeTrueFalsePerturbationTargets:
    def test_defaults_to_fair_thirds_when_possible(self):
        true_target, false_target = _compute_true_false_perturbation_targets(
            20,
            40,
            1 / 3,
            30,
        )

        assert true_target == 10
        assert false_target == 20

    def test_reallocates_when_true_examples_are_limited(self):
        true_target, false_target = _compute_true_false_perturbation_targets(
            3,
            40,
            1 / 3,
            30,
        )

        assert true_target == 3
        assert false_target == 27

    def test_invalid_true_fraction_raises(self):
        with pytest.raises(ValueError, match="true_perturbation_fraction"):
            _compute_true_false_perturbation_targets(5, 5, 1.5, 6)


# ── enforce_coupling ────────────────────────────────────────────


class TestEnforceCoupling:
    def test_adds_missing_originals(self):
        original = _make_sample(
            text="original",
            claim_id="c1",
            sample_type="Original",
            is_perturbed=False,
        )
        perturbed = _make_sample(
            text="perturbed",
            claim_id="c1",
            sample_type="T→T",
            is_perturbed=True,
        )
        # Selected has perturbation but not original
        selected = [perturbed]
        all_pool = [original, perturbed]

        result = enforce_coupling(selected, all_pool)
        assert len(result) == 2
        original_in_result = [s for s in result if not s.is_perturbed]
        assert len(original_in_result) == 1
        assert original_in_result[0].source_claim_id == "c1"

    def test_no_coupling_needed(self):
        original = _make_sample(text="orig", claim_id="c1", is_perturbed=False)
        perturbed = _make_sample(text="pert", claim_id="c1", is_perturbed=True)
        selected = [original, perturbed]

        result = enforce_coupling(selected, selected)
        assert len(result) == 2

    def test_originals_only_no_change(self):
        samples = [
            _make_sample(text="a", claim_id="c1", is_perturbed=False),
            _make_sample(text="b", claim_id="c2", is_perturbed=False),
        ]
        result = enforce_coupling(samples, samples)
        assert len(result) == 2


# ── apply_balancing ─────────────────────────────────────────────


class TestApplyBalancing:
    def _make_balanced_set(self):
        """Helper: 4 True originals, 4 False originals,
        4 T→T perturbed, 4 T→F perturbed, 4 F→F perturbed."""
        samples = []
        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"orig_t_{i}",
                    label="True",
                    claim_id=f"ct_{i}",
                    sample_type="Original",
                    is_perturbed=False,
                    source_file="file_a.json",
                )
            )
            samples.append(
                _make_sample(
                    text=f"orig_f_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    sample_type="Original",
                    is_perturbed=False,
                    source_file="file_b.json",
                )
            )
            samples.append(
                _make_sample(
                    text=f"tt_{i}",
                    label="True",
                    claim_id=f"ct_{i}",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                    source_file="file_a.json",
                )
            )
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    claim_id=f"ct_{i}",
                    sample_type="T→F",
                    method="approximation",
                    is_perturbed=True,
                    source_file="file_a.json",
                )
            )
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    sample_type="F→F",
                    method="range",
                    is_perturbed=True,
                    source_file="file_b.json",
                )
            )
        return samples

    def test_label_balancing(self):
        import random

        rng = random.Random(42)
        # More True than False
        samples = [
            _make_sample(text=f"t_{i}", label="True", is_perturbed=False)
            for i in range(10)
        ] + [
            _make_sample(text=f"f_{i}", label="False", is_perturbed=False)
            for i in range(4)
        ]

        result = apply_balancing(
            samples,
            balance_labels=True,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        true_count = sum(1 for s in result if s.label == "True")
        false_count = sum(1 for s in result if s.label == "False")
        assert true_count == false_count == 4

    def test_no_label_balancing(self):
        import random

        rng = random.Random(42)
        samples = [
            _make_sample(text=f"t_{i}", label="True", is_perturbed=False)
            for i in range(10)
        ] + [
            _make_sample(text=f"f_{i}", label="False", is_perturbed=False)
            for i in range(4)
        ]

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        assert len(result) == 14

    def test_perturb_ratio(self):
        import random

        rng = random.Random(42)
        samples = [
            _make_sample(
                text=f"orig_{i}",
                label="True",
                is_perturbed=False,
            )
            for i in range(10)
        ] + [
            _make_sample(
                text=f"pert_{i}",
                label="True",
                sample_type="T→T",
                method="numeration",
                is_perturbed=True,
            )
            for i in range(20)
        ]

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        originals = [s for s in result if not s.is_perturbed]
        perturbed = [s for s in result if s.is_perturbed]
        assert len(originals) == 10
        assert len(perturbed) == 10  # ratio 1.0

    def test_default_fair_split_between_transition_types(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(9):
            samples.append(
                _make_sample(
                    text=f"orig_{i}",
                    label="True" if i % 2 == 0 else "False",
                    claim_id=f"c_{i}",
                    is_perturbed=False,
                )
            )
            samples.append(
                _make_sample(
                    text=f"tt_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    sample_type="F→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=1.0,
            rng=rng,
        )

        perturbed = [s for s in result if s.is_perturbed]
        assert len([s for s in perturbed if s.type == "T→T"]) == 3
        assert len([s for s in perturbed if s.type == "T→F"]) == 3
        assert len([s for s in perturbed if s.type == "F→F"]) == 3

    def test_true_perturbation_fraction_customizes_true_budget(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(8):
            samples.append(
                _make_sample(
                    text=f"orig_{i}",
                    label="True" if i % 2 == 0 else "False",
                    claim_id=f"c_{i}",
                    is_perturbed=False,
                )
            )
        for i in range(8):
            samples.append(
                _make_sample(
                    text=f"tt_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    sample_type="F→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=1.0,
            true_perturbation_fraction=0.75,
            rng=rng,
        )

        perturbed = [s for s in result if s.is_perturbed]
        assert len([s for s in perturbed if s.type == "T→T"]) == 6
        assert len([s for s in perturbed if s.type == "T→F"]) == 1
        assert len([s for s in perturbed if s.type == "F→F"]) == 1

    def test_method_balancing(self):
        import random

        rng = random.Random(42)
        samples = []
        # Create imbalanced methods: 10 numeration, 5 approx, 3 range
        for i in range(10):
            samples.append(
                _make_sample(
                    text=f"num_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"approx_{i}",
                    label="True",
                    sample_type="T→T",
                    method="approximation",
                    is_perturbed=True,
                )
            )
        for i in range(3):
            samples.append(
                _make_sample(
                    text=f"range_{i}",
                    label="True",
                    sample_type="T→T",
                    method="range",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,  # high ratio, won't limit
            false_perturb_t_to_f_ratio=None,
            equal_method_size=True,
            method_balance_mode=None,
            rng=rng,
        )
        method_counts = Counter(s.perturbation_method for s in result)
        # All methods balanced to minimum (3)
        assert method_counts["numeration"] == 3
        assert method_counts["approximation"] == 3
        assert method_counts["range"] == 3

    def test_method_balancing_oversample(self):
        import random

        rng = random.Random(42)
        samples = []
        for i in range(10):
            samples.append(
                _make_sample(
                    text=f"num_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"approx_{i}",
                    label="True",
                    sample_type="T→T",
                    method="approximation",
                    is_perturbed=True,
                )
            )
        for i in range(3):
            samples.append(
                _make_sample(
                    text=f"range_{i}",
                    label="True",
                    sample_type="T→T",
                    method="range",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode="oversample",
            rng=rng,
        )
        method_counts = Counter(s.perturbation_method for s in result)
        assert method_counts["numeration"] == 10
        assert method_counts["approximation"] == 10
        assert method_counts["range"] == 10

    def test_method_balancing_undersample_false_types_independently(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tf_range_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"tf_masked_{i}",
                    label="False",
                    sample_type="T→F",
                    method="masked",
                    is_perturbed=True,
                )
            )
        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"ff_approx_{i}",
                    label="False",
                    sample_type="F→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )
        for i in range(1):
            samples.append(
                _make_sample(
                    text=f"ff_random_{i}",
                    label="False",
                    sample_type="F→F",
                    method="random",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode="undersample",
            rng=rng,
        )

        t_to_f_counts = Counter(
            s.perturbation_method for s in result if s.type == "T→F"
        )
        f_to_f_counts = Counter(
            s.perturbation_method for s in result if s.type == "F→F"
        )

        assert t_to_f_counts == Counter({"range": 2, "masked": 2})
        assert f_to_f_counts == Counter({"approximation": 1, "random": 1})

    def test_method_balancing_oversample_false_types_independently(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tf_range_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"tf_masked_{i}",
                    label="False",
                    sample_type="T→F",
                    method="masked",
                    is_perturbed=True,
                )
            )
        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"ff_approx_{i}",
                    label="False",
                    sample_type="F→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )
        for i in range(1):
            samples.append(
                _make_sample(
                    text=f"ff_random_{i}",
                    label="False",
                    sample_type="F→F",
                    method="random",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode="oversample",
            rng=rng,
        )

        t_to_f_counts = Counter(
            s.perturbation_method for s in result if s.type == "T→F"
        )
        f_to_f_counts = Counter(
            s.perturbation_method for s in result if s.type == "F→F"
        )

        assert t_to_f_counts == Counter({"range": 5, "masked": 5})
        assert f_to_f_counts == Counter({"approximation": 4, "random": 4})

    def test_method_oversample_respects_perturb_ratio(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"orig_{i}",
                    label="True",
                    claim_id=f"c_{i}",
                    is_perturbed=False,
                )
            )

        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"num_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )

        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"neg_{i}",
                    label="True",
                    sample_type="T→T",
                    method="negative",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=2.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            rng=rng,
            method_balance_mode="oversample",
        )

        originals = [s for s in result if not s.is_perturbed]
        perturbed = [s for s in result if s.is_perturbed]
        assert len(originals) == 4
        assert len(perturbed) == 8

    def test_method_balancing_none_leaves_counts_unchanged(self):
        import random

        rng = random.Random(42)
        samples = []
        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"num_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"neg_{i}",
                    label="True",
                    sample_type="T→T",
                    method="negative",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode="none",
            rng=rng,
        )
        method_counts = Counter(s.perturbation_method for s in result)
        assert method_counts["numeration"] == 4
        assert method_counts["negative"] == 2

    def test_invalid_method_balance_mode_raises(self):
        import random

        rng = random.Random(42)
        samples = [
            _make_sample(
                text="num_0",
                label="True",
                sample_type="T→T",
                method="numeration",
                is_perturbed=True,
            )
        ]

        with pytest.raises(ValueError, match="method_balance_mode"):
            apply_balancing(
                samples,
                balance_labels=False,
                perturb_ratio=100.0,
                false_perturb_t_to_f_ratio=None,
                equal_method_size=False,
                method_balance_mode="bad-mode",
                rng=rng,
            )

    def test_equal_method_size_conflicts_with_oversample(self):
        import random

        rng = random.Random(42)
        samples = [
            _make_sample(
                text="num_0",
                label="True",
                sample_type="T→T",
                method="numeration",
                is_perturbed=True,
            )
        ]

        with pytest.raises(ValueError, match="equal_method_size=True"):
            apply_balancing(
                samples,
                balance_labels=False,
                perturb_ratio=100.0,
                false_perturb_t_to_f_ratio=None,
                equal_method_size=True,
                method_balance_mode="oversample",
                rng=rng,
            )

    def test_method_oversample_respects_ratio_and_false_composition(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(4):
            samples.append(
                _make_sample(
                    text=f"orig_t_{i}",
                    label="True",
                    claim_id=f"ct_{i}",
                    is_perturbed=False,
                )
            )
            samples.append(
                _make_sample(
                    text=f"orig_f_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    is_perturbed=False,
                )
            )

        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tt_num_{i}",
                    label="True",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"tt_neg_{i}",
                    label="True",
                    sample_type="T→T",
                    method="negative",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tf_range_{i}",
                    label="False",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"tf_masked_{i}",
                    label="False",
                    sample_type="T→F",
                    method="masked",
                    is_perturbed=True,
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"ff_approx_{i}",
                    label="False",
                    sample_type="F→F",
                    method="approximation",
                    is_perturbed=True,
                )
            )
        for i in range(2):
            samples.append(
                _make_sample(
                    text=f"ff_random_{i}",
                    label="False",
                    sample_type="F→F",
                    method="random",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=2.0,
            false_perturb_t_to_f_ratio=0.5,
            equal_method_size=False,
            method_balance_mode="oversample",
            rng=rng,
        )

        originals = [s for s in result if not s.is_perturbed]
        perturbed = [s for s in result if s.is_perturbed]
        true_perturbed = [s for s in perturbed if s.type == "T→T"]
        t_to_f = [s for s in perturbed if s.type == "T→F"]
        f_to_f = [s for s in perturbed if s.type == "F→F"]

        assert len(originals) == 8
        assert len(perturbed) == 16
        assert len(true_perturbed) == 5
        assert len(t_to_f) + len(f_to_f) == 11
        assert abs(len(t_to_f) - len(f_to_f)) <= 1

        t_to_f_counts = Counter(s.perturbation_method for s in t_to_f)
        f_to_f_counts = Counter(s.perturbation_method for s in f_to_f)

        assert max(t_to_f_counts.values()) - min(t_to_f_counts.values()) <= 1
        assert max(f_to_f_counts.values()) - min(f_to_f_counts.values()) <= 1

    def test_false_perturb_composition(self):
        import random

        rng = random.Random(42)
        samples = []
        # 20 T→F, 10 F→F
        for i in range(20):
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    claim_id=f"ct_{i}",
                    sample_type="T→F",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(10):
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    sample_type="F→F",
                    method="range",
                    is_perturbed=True,
                )
            )

        # ratio=0.5 means 50% T→F, 50% F→F → undersample T→F to 10
        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=0.5,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        t_to_f = [s for s in result if s.type == "T→F"]
        f_to_f = [s for s in result if s.type == "F→F"]
        assert len(t_to_f) == 10
        assert len(f_to_f) == 10

    def test_false_perturb_ratio_undersamples_f_to_f(self):
        import random

        rng = random.Random(42)
        samples = []
        # 5 T→F, 20 F→F
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    claim_id=f"ct_{i}",
                    sample_type="T→F",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(20):
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    sample_type="F→F",
                    method="range",
                    is_perturbed=True,
                )
            )

        # ratio=0.5 → F→F undersampled to 5
        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=0.5,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        t_to_f = [s for s in result if s.type == "T→F"]
        f_to_f = [s for s in result if s.type == "F→F"]
        assert len(t_to_f) == 5
        assert len(f_to_f) == 5

    def test_false_perturb_ratio_none_no_constraint(self):
        import random

        rng = random.Random(42)
        samples = []
        # 5 T→F, 20 F→F
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    claim_id=f"ct_{i}",
                    sample_type="T→F",
                    method="numeration",
                    is_perturbed=True,
                )
            )
        for i in range(20):
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    claim_id=f"cf_{i}",
                    sample_type="F→F",
                    method="range",
                    is_perturbed=True,
                )
            )

        # None → no constraint, both untouched
        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )
        t_to_f = [s for s in result if s.type == "T→F"]
        f_to_f = [s for s in result if s.type == "F→F"]
        assert len(t_to_f) == 5
        assert len(f_to_f) == 20

    def test_false_perturb_fraction_respected_with_budget(self):
        import random

        rng = random.Random(42)
        samples = []

        for i in range(50):
            samples.append(
                _make_sample(
                    text=f"orig_{i}",
                    label="True" if i % 2 == 0 else "False",
                    claim_id=f"co_{i}",
                )
            )

        for i in range(20):
            samples.append(
                _make_sample(
                    text=f"tt_{i}",
                    label="True",
                    claim_id=f"ctt_{i}",
                    sample_type="T→T",
                    method="numeration",
                    is_perturbed=True,
                )
            )

        for i in range(40):
            samples.append(
                _make_sample(
                    text=f"tf_{i}",
                    label="False",
                    claim_id=f"ctf_{i}",
                    sample_type="T→F",
                    method="range",
                    is_perturbed=True,
                )
            )

        for i in range(80):
            samples.append(
                _make_sample(
                    text=f"ff_{i}",
                    label="False",
                    claim_id=f"cff_{i}",
                    sample_type="F→F",
                    method="numeration",
                    is_perturbed=True,
                )
            )

        result = apply_balancing(
            samples,
            balance_labels=False,
            perturb_ratio=1.0,
            false_perturb_t_to_f_ratio=0.1,
            equal_method_size=False,
            method_balance_mode=None,
            rng=rng,
        )

        perturbed = [s for s in result if s.is_perturbed]
        t_to_t = [s for s in perturbed if s.type == "T→T"]
        t_to_f = [s for s in perturbed if s.type == "T→F"]
        f_to_f = [s for s in perturbed if s.type == "F→F"]

        assert len(perturbed) == 50
        assert len(t_to_t) == 17
        assert len(t_to_f) == 3
        assert len(f_to_f) == 30


# ── build_balanced_dataset (integration) ────────────────────────


class TestBuildBalancedDataset:
    def _create_test_files(self, tmp_path):
        """Create test JSON files simulating real data."""
        true_records = []
        for i in range(10):
            true_records.append(
                _make_record(
                    claim_id=f"claim_{i:04d}",
                    label="True",
                    normalized_claim=f"True claim number {i}.",
                    method="numeration",
                    perturb_fields={
                        "perturb_type1_True": (f"True claim number {i} type1."),
                        "perturb_type2_True": (f"True claim number {i} type2."),
                    },
                )
            )

        false_records = []
        for i in range(10, 20):
            false_records.append(
                _make_record(
                    claim_id=f"claim_{i:04d}",
                    label="False",
                    normalized_claim=f"False claim number {i}.",
                    method="approximation",
                    perturb_fields={
                        "perturb_type1_False": (
                            f"False claim number {i} type1."
                        ),
                        "perturb_type2_False": (
                            f"False claim number {i} type2."
                        ),
                    },
                )
            )

        path_t = _write_json(true_records, tmp_path, "numeration_True.json")
        path_f = _write_json(
            false_records, tmp_path, "approximation_False.json"
        )
        return [path_t, path_f]

    def test_basic_output_schema(self, tmp_path):
        paths = self._create_test_files(tmp_path)
        result = build_balanced_dataset(paths, balance_labels=False, seed=42)
        assert len(result) > 0
        sample = result[0]
        assert set(sample.keys()) == {
            "text",
            "label",
            "source_claim_id",
            "type",
            "perturbation_method",
            "is_perturbed",
            "source_file",
        }

    def test_deterministic_with_seed(self, tmp_path):
        paths = self._create_test_files(tmp_path)
        r1 = build_balanced_dataset(paths, seed=42)
        r2 = build_balanced_dataset(paths, seed=42)
        assert r1 == r2

    def test_different_seed_different_order(self, tmp_path):
        paths = self._create_test_files(tmp_path)
        kwargs = dict(
            balance_labels=False,
            perturb_ratio=100.0,
            false_perturb_t_to_f_ratio=None,  # no constraint
        )
        r1 = build_balanced_dataset(paths, seed=42, **kwargs)
        r2 = build_balanced_dataset(paths, seed=99, **kwargs)
        # Same samples, different order (almost certainly)
        assert sorted(s["text"] for s in r1) == sorted(s["text"] for s in r2)

    def test_label_balance(self, tmp_path):
        paths = self._create_test_files(tmp_path)
        result = build_balanced_dataset(paths, balance_labels=True, seed=42)
        labels = Counter(s["label"] for s in result)
        # Coupling (priority 2) can add originals after label
        # balancing (priority 3), so allow a small delta.
        assert abs(labels["True"] - labels["False"]) <= 2

    def test_no_duplicate_texts(self, tmp_path):
        # Create files with overlapping claims
        recs_a = [
            _make_record("c1", "True", "Shared claim."),
            _make_record("c2", "True", "Unique A."),
        ]
        recs_b = [
            _make_record("c3", "True", "Shared claim."),
            _make_record("c4", "True", "Unique B."),
        ]
        pa = _write_json(recs_a, tmp_path, "a.json")
        pb = _write_json(recs_b, tmp_path, "b.json")

        result = build_balanced_dataset([pa, pb], balance_labels=False, seed=42)
        texts = [s["text"] for s in result]
        assert len(texts) == len(set(texts))

    def test_coupling_preserved(self, tmp_path):
        """When perturbation is included, original must be too."""
        recs = [
            _make_record(
                "c1",
                "True",
                "The claim.",
                perturb_fields={
                    "perturb_type1_True": "Perturbed claim.",
                },
            )
        ]
        path = _write_json(recs, tmp_path, "data.json")

        result = build_balanced_dataset([path], balance_labels=False, seed=42)

        claim_ids_perturbed = {
            s["source_claim_id"] for s in result if s["is_perturbed"]
        }
        claim_ids_original = {
            s["source_claim_id"] for s in result if not s["is_perturbed"]
        }
        # Every perturbed claim_id has its original
        assert claim_ids_perturbed <= claim_ids_original

    def test_claims_without_perturbations_are_valid(self, tmp_path):
        recs = [
            _make_record("c1", "True", "Simple claim."),
        ]
        path = _write_json(recs, tmp_path, "data.json")
        result = build_balanced_dataset([path], balance_labels=False, seed=42)
        assert len(result) == 1
        assert result[0]["type"] == "Original"

    def test_perturb_ratio_enforcement(self, tmp_path):
        recs = []
        for i in range(5):
            recs.append(
                _make_record(
                    f"c{i}",
                    "True",
                    f"Claim {i}.",
                    perturb_fields={
                        "perturb_type1_True": f"Perturbed {i} t1.",
                        "perturb_type2_True": f"Perturbed {i} t2.",
                    },
                )
            )
        path = _write_json(recs, tmp_path, "data.json")

        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=1.0,
            seed=42,
        )
        originals = [s for s in result if not s["is_perturbed"]]
        perturbed = [s for s in result if s["is_perturbed"]]
        # With ratio 1.0, perturbed <= originals
        assert len(perturbed) <= len(originals) + 1

    def test_method_balance_mode_oversample(self, tmp_path):
        records = []
        for i in range(5):
            records.append(
                _make_record(
                    claim_id=f"num_{i}",
                    label="True",
                    normalized_claim=f"Numeration claim {i}.",
                    method="numeration",
                    perturb_fields={
                        "perturb_type1_True": f"Numeration perturb {i}.",
                    },
                )
            )
        for i in range(2):
            records.append(
                _make_record(
                    claim_id=f"neg_{i}",
                    label="True",
                    normalized_claim=f"Negative claim {i}.",
                    method="negative",
                    perturb_fields={
                        "perturb_type1_True": f"Negative perturb {i}.",
                    },
                )
            )

        path = _write_json(records, tmp_path, "methods.json")
        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=100.0,
            method_balance_mode="oversample",
            seed=42,
        )

        method_counts = Counter(
            s["perturbation_method"] for s in result if s["is_perturbed"]
        )
        assert method_counts["numeration"] == 5
        assert method_counts["negative"] == 5

    def test_method_balance_mode_undersample(self, tmp_path):
        records = []
        for i in range(5):
            records.append(
                _make_record(
                    claim_id=f"num_{i}",
                    label="True",
                    normalized_claim=f"Numeration claim {i}.",
                    method="numeration",
                    perturb_fields={
                        "perturb_type1_True": f"Numeration perturb {i}.",
                    },
                )
            )
        for i in range(2):
            records.append(
                _make_record(
                    claim_id=f"neg_{i}",
                    label="True",
                    normalized_claim=f"Negative claim {i}.",
                    method="negative",
                    perturb_fields={
                        "perturb_type1_True": f"Negative perturb {i}.",
                    },
                )
            )

        path = _write_json(records, tmp_path, "methods.json")
        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=100.0,
            method_balance_mode="undersample",
            seed=42,
        )

        method_counts = Counter(
            s["perturbation_method"] for s in result if s["is_perturbed"]
        )
        assert method_counts["numeration"] == 2
        assert method_counts["negative"] == 2

    def test_method_balance_mode_none(self, tmp_path):
        records = []
        for i in range(5):
            records.append(
                _make_record(
                    claim_id=f"num_{i}",
                    label="True",
                    normalized_claim=f"Numeration claim {i}.",
                    method="numeration",
                    perturb_fields={
                        "perturb_type1_True": f"Numeration perturb {i}.",
                    },
                )
            )
        for i in range(2):
            records.append(
                _make_record(
                    claim_id=f"neg_{i}",
                    label="True",
                    normalized_claim=f"Negative claim {i}.",
                    method="negative",
                    perturb_fields={
                        "perturb_type1_True": f"Negative perturb {i}.",
                    },
                )
            )

        path = _write_json(records, tmp_path, "methods.json")
        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=100.0,
            method_balance_mode="none",
            seed=42,
        )

        method_counts = Counter(
            s["perturbation_method"] for s in result if s["is_perturbed"]
        )
        assert method_counts["numeration"] == 5
        assert method_counts["negative"] == 2


# ── compute_stats ───────────────────────────────────────────────


class TestComputeStats:
    def test_stats_structure(self):
        data = [
            {
                "text": "a",
                "label": "True",
                "source_claim_id": "c1",
                "type": "Original",
                "perturbation_method": None,
                "is_perturbed": False,
                "source_file": "f.json",
            },
            {
                "text": "b",
                "label": "False",
                "source_claim_id": "c1",
                "type": "T→F",
                "perturbation_method": "numeration",
                "is_perturbed": True,
                "source_file": "f.json",
            },
        ]
        stats = compute_stats(data)
        assert stats["total_samples"] == 2
        assert stats["label_distribution"] == {
            "True": 1,
            "False": 1,
        }
        assert stats["type_distribution"] == {
            "Original": 1,
            "T→F": 1,
        }
        assert stats["method_distribution"] == {"numeration": 1}
        assert stats["file_distribution"] == {"f.json": 2}


# ── Edge cases / integration with real data schema ──────────────


class TestEdgeCases:
    def test_multi_file_fairness(self, tmp_path):
        """Samples should be drawn proportionally from files."""
        recs_a = [
            _make_record(f"a{i}", "True", f"Claim A{i}.") for i in range(20)
        ]
        recs_b = [
            _make_record(f"b{i}", "True", f"Claim B{i}.") for i in range(20)
        ]
        pa = _write_json(recs_a, tmp_path, "a.json")
        pb = _write_json(recs_b, tmp_path, "b.json")

        result = build_balanced_dataset([pa, pb], balance_labels=False, seed=42)
        file_counts = Counter(s["source_file"] for s in result)
        # Both files contribute equally
        assert file_counts["a.json"] == file_counts["b.json"]

    def test_obfuscation_type3_fields(self, tmp_path):
        recs = [
            _make_record(
                "c1",
                "True",
                "Number 100 here.",
                method="random",
                perturb_fields={
                    "perturb_type3_random_True": "Number ### here.",
                    "perturb_type3_masked_True": "Number XXX here.",
                    "perturb_type3_removed_True": "Number here.",
                },
            )
        ]
        path = _write_json(recs, tmp_path, "obfuscation.json")
        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=10.0,
            seed=42,
        )
        perturbed = [s for s in result if s["is_perturbed"]]
        # "removed" excluded, so only random + masked
        assert len(perturbed) == 2
        texts = {s["text"] for s in perturbed}
        assert "Number ### here." in texts
        assert "Number XXX here." in texts
        assert "Number here." not in texts
        for s in perturbed:
            assert s["label"] == "False"
            assert s["type"] == "T→F"

    def test_removed_method_excluded_entirely(self, tmp_path):
        """Records with pertubation_method='removed' are skipped."""
        recs = [
            _make_record(
                "c1",
                "True",
                "Claim from removed method.",
                method="removed",
                perturb_fields={
                    "perturb_type3_removed_True": "Removed text.",
                },
            ),
            _make_record(
                "c2",
                "True",
                "Claim from random method.",
                method="random",
                perturb_fields={
                    "perturb_type3_random_True": "Random text.",
                },
            ),
        ]
        path = _write_json(recs, tmp_path, "mixed.json")
        result = build_balanced_dataset(
            [path],
            balance_labels=False,
            perturb_ratio=10.0,
            seed=42,
        )
        # c1 is entirely excluded (method=removed)
        claim_ids = {s["source_claim_id"] for s in result}
        assert "c1" not in claim_ids
        assert "c2" in claim_ids

    def test_negative_perturbation(self, tmp_path):
        recs = [
            _make_record(
                "c1",
                "True",
                "Value is 10.",
                method="negative",
                perturb_fields={
                    "perturb_type2_True": "Value is -10.",
                },
            )
        ]
        path = _write_json(recs, tmp_path, "negative.json")
        result = build_balanced_dataset([path], balance_labels=False, seed=42)
        perturbed = [s for s in result if s["is_perturbed"]]
        assert len(perturbed) == 1
        assert perturbed[0]["label"] == "False"
        assert perturbed[0]["type"] == "T→F"

    def test_balancing_removes_all_perturbs_keeps_original(self, tmp_path):
        """If balancing removes all perturbations of a claim, the original can
        remain independently."""
        # Create imbalanced: many True, few False
        recs = [
            _make_record(
                "c1",
                "True",
                "True claim one.",
                perturb_fields={
                    "perturb_type1_True": "Perturbed T→T.",
                },
            ),
            _make_record("c2", "False", "False claim one."),
        ]
        path = _write_json(recs, tmp_path, "data.json")
        result = build_balanced_dataset([path], balance_labels=True, seed=42)
        # Should still have originals even if perturbations removed
        originals = [s for s in result if not s["is_perturbed"]]
        assert len(originals) >= 1
