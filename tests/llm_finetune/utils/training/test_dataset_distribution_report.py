"""Tests for src.llm_finetune.utils.training.dataset_distribution_report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from src.llm_finetune.utils.training.dataset_distribution_report import (
    DISTRIBUTION_REPORT_FILENAME,
    ORIGINAL_NORMALIZED,
    _counter_table_lines,
    _parse_label,
    compute_distribution,
    format_distribution_report,
    save_distribution_report,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_dataset(
    n: int = 12,
    perturb_types: list[str] | None = None,
    labels: list[str] | None = None,
    source_files: list[str] | None = None,
    methods: list[str] | None = None,
    sample_types: list[str] | None = None,
) -> Dataset:
    """Build a minimal dataset for testing distribution reporting.

    Args:
        n: Number of examples.
        perturb_types: Types to cycle through.
        labels: Labels to cycle through (plain or JSON format).
        source_files: Source file names to cycle through.
        methods: Method strings to cycle through.
        sample_types: Directional sample types to cycle through.

    Returns:
        HuggingFace Dataset with the expected columns.
    """
    if perturb_types is None:
        perturb_types = [
            ORIGINAL_NORMALIZED,
            "perturb_type1",
            "perturb_type2",
        ]
    if labels is None:
        labels = ['{"label": true}', '{"label": false}']
    if source_files is None:
        source_files = ["approx_True.json", "numeration_False.json"]
    if methods is None:
        methods = ["", "approximation", "numeration"]
    if sample_types is None:
        sample_types = ["Original", "T\u2192T", "T\u2192F", "F\u2192F"]

    return Dataset.from_dict(
        {
            "claim_id": [f"claim_{i}" for i in range(n)],
            "perturbation_type": [
                perturb_types[i % len(perturb_types)] for i in range(n)
            ],
            "label": [labels[i % len(labels)] for i in range(n)],
            "source_file": [
                source_files[i % len(source_files)] for i in range(n)
            ],
            "perturbation_method": [
                methods[i % len(methods)] for i in range(n)
            ],
            "sample_type": [
                sample_types[i % len(sample_types)] for i in range(n)
            ],
        }
    )


# ------------------------------------------------------------------
# _parse_label
# ------------------------------------------------------------------


class TestParseLabel:
    """Tests for the internal label normaliser."""

    def test_json_true(self):
        """JSON-format true label is parsed as 'True'."""
        assert _parse_label('{"label": true}') == "True"

    def test_json_false(self):
        """JSON-format false label is parsed as 'False'."""
        assert _parse_label('{"label": false}') == "False"

    def test_plain_true(self):
        """Plain 'True' string is returned as-is."""
        assert _parse_label("True") == "True"

    def test_plain_false(self):
        """Plain 'False' string is returned as-is."""
        assert _parse_label("False") == "False"

    def test_case_insensitive_true(self):
        """Lower-case 'true' is normalised to 'True'."""
        assert _parse_label("true") == "True"

    def test_case_insensitive_false(self):
        """Lower-case 'false' is normalised to 'False'."""
        assert _parse_label("false") == "False"

    def test_unknown_value(self):
        """Unknown values are returned as their string representation."""
        assert _parse_label("UNKNOWN") == "UNKNOWN"


# ------------------------------------------------------------------
# _counter_table_lines
# ------------------------------------------------------------------


class TestCounterTableLines:
    """Tests for _counter_table_lines."""

    def test_returns_one_line_per_entry(self):
        """One formatted line is produced per counter entry."""
        from collections import Counter

        ctr = Counter({"a": 3, "b": 1})
        lines = _counter_table_lines(ctr, total=4)
        assert len(lines) == 2

    def test_percentage_calculation(self):
        """Percentages in the output reflect the correct ratio."""
        from collections import Counter

        ctr = Counter({"x": 1, "y": 1})
        lines = _counter_table_lines(ctr, total=2)
        # Both entries should show 50.0%
        assert all("50.0%" in line for line in lines)

    def test_zero_total_no_division_error(self):
        """A total of zero does not raise a ZeroDivisionError."""
        from collections import Counter

        ctr = Counter({"k": 5})
        lines = _counter_table_lines(ctr, total=0)
        assert len(lines) == 1


# ------------------------------------------------------------------
# compute_distribution
# ------------------------------------------------------------------


class TestComputeDistribution:
    """Tests for compute_distribution."""

    def test_empty_dataset(self):
        """An empty dataset returns all-zero stats."""
        ds = Dataset.from_dict(
            {
                "perturbation_type": [],
                "label": [],
                "source_file": [],
                "perturbation_method": [],
                "sample_type": [],
            }
        )
        stats = compute_distribution(ds)

        assert stats["total"] == 0
        assert stats["originals"] == 0
        assert stats["perturbed"] == 0
        assert stats["perturb_to_orig_ratio"] is None
        assert stats["label_distribution"] == {}
        assert stats["type_distribution"] == {}
        assert stats["perturbation_class_composition"]["T2T"] == 0
        assert stats["false_pert_composition"]["T2F_ratio"] is None
        assert stats["method_x_type_crosstab"] == {}

    def test_total_matches_dataset_size(self):
        """Total count matches the number of examples in the dataset."""
        ds = _make_dataset(n=12)
        stats = compute_distribution(ds)
        assert stats["total"] == 12

    def test_original_vs_perturbed_counts(self):
        """Origins and perturbed counts sum to the total."""
        ds = _make_dataset(
            n=9, perturb_types=[ORIGINAL_NORMALIZED] * 3 + ["perturb_type1"] * 6
        )
        stats = compute_distribution(ds)
        assert stats["originals"] + stats["perturbed"] == stats["total"]

    def test_all_originals(self):
        """Dataset of only originals has zero perturbed and null ratio."""
        ds = _make_dataset(
            n=4,
            perturb_types=[ORIGINAL_NORMALIZED],
        )
        stats = compute_distribution(ds)
        assert stats["originals"] == 4
        assert stats["perturbed"] == 0
        assert stats["perturb_to_orig_ratio"] == pytest.approx(0.0)

    def test_perturb_to_orig_ratio(self):
        """Ratio equals perturbed / originals."""
        ds = _make_dataset(
            n=6,
            perturb_types=[ORIGINAL_NORMALIZED, "perturb_type1"],
        )
        stats = compute_distribution(ds)
        assert stats["perturb_to_orig_ratio"] == pytest.approx(
            stats["perturbed"] / stats["originals"]
        )

    def test_label_distribution_json_format(self):
        """JSON-format labels are parsed and counted correctly."""
        ds = _make_dataset(
            n=4,
            labels=['{"label": true}', '{"label": false}'],
        )
        stats = compute_distribution(ds)
        label_dist = stats["label_distribution"]
        assert label_dist.get("True", 0) + label_dist.get("False", 0) == 4

    def test_label_distribution_plain_format(self):
        """Plain 'True'/'False' labels are counted correctly."""
        ds = _make_dataset(n=6, labels=["True", "False", "True"])
        stats = compute_distribution(ds)
        assert stats["label_distribution"]["True"] == 4
        assert stats["label_distribution"]["False"] == 2

    def test_perturbation_type_distribution_present(self):
        """All perturbation types appear in the type distribution."""
        ds = _make_dataset(
            n=6,
            perturb_types=[ORIGINAL_NORMALIZED, "perturb_type1"],
        )
        stats = compute_distribution(ds)
        assert ORIGINAL_NORMALIZED in stats["perturbation_type_distribution"]
        assert "perturb_type1" in stats["perturbation_type_distribution"]

    def test_source_file_distribution(self):
        """Source file counts reflect the input data."""
        ds = _make_dataset(
            n=4,
            source_files=["file_a.json", "file_b.json"],
        )
        stats = compute_distribution(ds)
        assert "file_a.json" in stats["source_file_distribution"]
        assert "file_b.json" in stats["source_file_distribution"]

    def test_method_distribution(self):
        """Method counts are non-empty when methods are present."""
        ds = _make_dataset(
            n=6,
            methods=["approximation", "numeration"],
        )
        stats = compute_distribution(ds)
        assert "approximation" in stats["method_distribution"]

    def test_cross_tab_sums_to_total(self):
        """label_by_perturbation cross-tab entries sum to total."""
        ds = _make_dataset(n=12)
        stats = compute_distribution(ds)
        cross = stats["label_by_perturbation"]
        total = sum(cross.values())
        assert total == stats["total"]

    def test_cross_tab_original_true(self):
        """Original/True counts are separated from perturbed/True."""
        ds = _make_dataset(
            n=4,
            perturb_types=[ORIGINAL_NORMALIZED, "perturb_type1"],
            labels=['{"label": true}'],
        )
        stats = compute_distribution(ds)
        cross = stats["label_by_perturbation"]
        # All labels are True; split should match originals/perturbed counts
        assert cross["original_true"] == stats["originals"]
        assert cross["perturbed_true"] == stats["perturbed"]
        assert cross["original_false"] == 0
        assert cross["perturbed_false"] == 0

    def test_missing_columns_do_not_raise(self):
        """Missing optional columns return empty dicts without errors."""
        ds = Dataset.from_dict({"claim_id": ["c1", "c2"]})
        stats = compute_distribution(ds)
        assert stats["total"] == 2
        assert stats["label_distribution"] == {}
        assert stats["perturbation_type_distribution"] == {}
        assert stats["type_distribution"] == {}
        assert stats["method_x_type_crosstab"] == {}

    def test_type_distribution_keys(self):
        """type_distribution contains the expected directional type keys."""
        ds = _make_dataset(
            n=4,
            perturb_types=["perturb_type1"],
            sample_types=["T\u2192T", "T\u2192F", "F\u2192F", "T\u2192T"],
        )
        stats = compute_distribution(ds)
        td = stats["type_distribution"]
        assert td.get("T\u2192T", 0) == 2
        assert td.get("T\u2192F", 0) == 1
        assert td.get("F\u2192F", 0) == 1

    def test_perturbation_class_composition_counts(self):
        """T2T and false_pert counts sum to total perturbed."""
        ds = _make_dataset(
            n=6,
            perturb_types=[ORIGINAL_NORMALIZED, "perturb_type1"],
            sample_types=[
                "Original",
                "T\u2192T",
                "Original",
                "T\u2192F",
                "Original",
                "F\u2192F",
            ],
        )
        stats = compute_distribution(ds)
        pcc = stats["perturbation_class_composition"]
        assert pcc["T2T"] + pcc["false_pert"] == stats["perturbed"]
        assert pcc["false_pert"] == pcc["T2F"] + pcc["F2F"]

    def test_false_pert_composition_ratio(self):
        """T2F_ratio equals T2F / (T2F + F2F)."""
        ds = _make_dataset(
            n=4,
            perturb_types=["perturb_type1"],
            sample_types=["T\u2192F", "T\u2192F", "F\u2192F", "F\u2192F"],
        )
        stats = compute_distribution(ds)
        fpc = stats["false_pert_composition"]
        assert fpc["T2F"] == 2
        assert fpc["F2F"] == 2
        assert fpc["T2F_ratio"] == pytest.approx(0.5)

    def test_false_pert_ratio_none_when_no_false_pert(self):
        """T2F_ratio is None when there are no false perturbations."""
        ds = _make_dataset(
            n=4,
            perturb_types=[ORIGINAL_NORMALIZED],
            sample_types=["Original"],
        )
        stats = compute_distribution(ds)
        assert stats["false_pert_composition"]["T2F_ratio"] is None

    def test_method_x_type_crosstab_structure(self):
        """Cross-tab produces a nested dict keyed by method then type."""
        ds = _make_dataset(
            n=6,
            perturb_types=["perturb_type1"],
            methods=[
                "approximation",
                "numeration",
                "approximation",
                "numeration",
                "approximation",
                "numeration",
            ],
            sample_types=[
                "T\u2192T",
                "T\u2192F",
                "T\u2192T",
                "F\u2192F",
                "T\u2192T",
                "T\u2192F",
            ],
        )
        stats = compute_distribution(ds)
        ct = stats["method_x_type_crosstab"]
        assert "approximation" in ct
        assert "numeration" in ct
        assert ct["approximation"].get("T\u2192T", 0) == 3
        assert ct["numeration"].get("T\u2192F", 0) == 2

    def test_method_x_type_empty_method_keyed_as_original(self):
        """Rows with empty method string are grouped under '(original)'."""
        ds = _make_dataset(
            n=3,
            perturb_types=[ORIGINAL_NORMALIZED],
            methods=[""],
            sample_types=["Original"],
        )
        stats = compute_distribution(ds)
        ct = stats["method_x_type_crosstab"]
        assert "(original)" in ct
        assert ct["(original)"].get("Original", 0) == 3


# ------------------------------------------------------------------
# format_distribution_report
# ------------------------------------------------------------------


class TestFormatDistributionReport:
    """Tests for format_distribution_report."""

    def test_returns_string(self):
        """Output is a non-empty string."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_title_appears_in_output(self):
        """Custom title appears in the formatted report."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats, title="MY CUSTOM TITLE")
        assert "MY CUSTOM TITLE" in report

    def test_total_appears_in_output(self):
        """Total example count appears in the formatted report."""
        ds = _make_dataset(n=8)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "8" in report

    def test_ratio_null_shown_as_na(self):
        """A null ratio (no originals) is shown as 'N/A'."""
        stats = {
            "total": 0,
            "originals": 0,
            "perturbed": 0,
            "perturb_to_orig_ratio": None,
            "label_distribution": {},
            "perturbation_type_distribution": {},
            "source_file_distribution": {},
            "method_distribution": {},
            "label_by_perturbation": {},
        }
        report = format_distribution_report(stats)
        assert "N/A" in report

    def test_label_section_present(self):
        """The label distribution section appears in the output."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "Label distribution" in report

    def test_perturbation_type_section_present(self):
        """The method distribution section appears in the output."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "Perturbation method distribution" in report

    def test_type_distribution_section_present(self):
        """The type distribution section appears when sample_type present."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "Type distribution" in report

    def test_perturbation_class_section_present(self):
        """The perturbation class composition section appears in output."""
        ds = _make_dataset(
            n=6,
            perturb_types=[ORIGINAL_NORMALIZED, "perturb_type1"],
            sample_types=["Original", "T\u2192T"],
        )
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "Perturbation class composition" in report

    def test_false_pert_composition_section_present(self):
        """The false pert composition section appears when T→F or F→F exist."""
        ds = _make_dataset(
            n=4,
            perturb_types=["perturb_type1"],
            sample_types=["T\u2192F", "F\u2192F", "T\u2192F", "F\u2192F"],
        )
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "False perturbation composition" in report

    def test_crosstab_section_present(self):
        """The method × type cross-tab section appears in the output."""
        ds = _make_dataset(n=6)
        stats = compute_distribution(ds)
        report = format_distribution_report(stats)
        assert "Method" in report and "cross-tab" in report


# ------------------------------------------------------------------
# save_distribution_report
# ------------------------------------------------------------------


class TestSaveDistributionReport:
    """Tests for save_distribution_report."""

    def test_creates_file(self):
        """The output JSON file is created at the requested path."""
        ds = _make_dataset(n=6)
        train_stats = compute_distribution(ds)
        val_stats = compute_distribution(_make_dataset(n=3))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run" / DISTRIBUTION_REPORT_FILENAME
            returned = save_distribution_report(train_stats, val_stats, out)
            assert out.exists()
            assert returned == out

    def test_json_is_valid(self):
        """The written file contains valid JSON."""
        ds = _make_dataset(n=6)
        train_stats = compute_distribution(ds)
        val_stats = compute_distribution(_make_dataset(n=3))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dist.json"
            save_distribution_report(train_stats, val_stats, out)
            with open(out, encoding="utf-8") as fh:
                payload = json.load(fh)

        assert "train" in payload
        assert "val" in payload

    def test_train_total_persisted(self):
        """Train total in saved JSON matches the computed stats."""
        train_stats = compute_distribution(_make_dataset(n=10))
        val_stats = compute_distribution(_make_dataset(n=5))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dist.json"
            save_distribution_report(train_stats, val_stats, out)
            with open(out, encoding="utf-8") as fh:
                payload = json.load(fh)

        assert payload["train"]["total"] == 10
        assert payload["val"]["total"] == 5

    def test_parent_dirs_created(self):
        """Missing parent directories are created automatically."""
        train_stats = compute_distribution(_make_dataset(n=4))
        val_stats = compute_distribution(_make_dataset(n=2))

        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "a" / "b" / "c" / "dist.json"
            save_distribution_report(train_stats, val_stats, deep)
            assert deep.exists()
