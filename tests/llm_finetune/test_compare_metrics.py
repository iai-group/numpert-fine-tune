"""Tests for src/llm_finetune/model_compare/compare_metrics.py."""

import csv
import json
import math
from pathlib import Path

import pytest

from src.llm_finetune.model_compare.compare_metrics import (
    build_rows,
    discover_metrics_files,
    extract_normalized_baseline,
    extract_split_groups,
    infer_model_name,
    model_sort_key,
    safe_float,
    safe_float_or_nan,
    safe_int,
    split_groups_by_perturb_family,
    to_wide,
    write_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_metrics_file(directory: Path, model_name: str, data: dict) -> Path:
    """Write a metrics.json file inside directory / model_name /."""
    model_dir = directory / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "metrics.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def minimal_metrics(
    *,
    true_group: bool = True,
    false_group: bool = True,
    normalized: bool = False,
    overall_total: int = 100,
    overall_parse_failures: int = 5,
) -> dict:
    """Return a minimal metrics.json structure."""
    per_file_and_type: dict = {}
    if true_group:
        per_file_and_type["approximation_True.json | perturb_type1"] = {
            "accuracy": 0.8,
            "parse_failures": 2,
            "total": 50,
            "valid": 48,
            "correct": 40,
        }
    if false_group:
        per_file_and_type["approximation_False.json | perturb_type1"] = {
            "accuracy": 0.6,
            "parse_failures": 1,
            "total": 50,
            "valid": 49,
            "correct": 30,
        }

    per_perturbation_type: dict = {}
    if normalized:
        per_perturbation_type["original_normalized"] = {
            "accuracy": 0.75,
            "parse_failures": 3,
            "total": 80,
            "valid": 77,
            "correct": 60,
        }

    return {
        "overall": {
            "total": overall_total,
            "parse_failures": overall_parse_failures,
        },
        "per_file_and_type": per_file_and_type,
        "per_perturbation_type": per_perturbation_type,
    }


# ---------------------------------------------------------------------------
# model_sort_key
# ---------------------------------------------------------------------------


class TestModelSortKey:
    def test_baseline_sorts_first(self):
        assert model_sort_key("baseline_model")[0] == 0

    def test_normalized_sorts_second(self):
        assert model_sort_key("normalized_model")[0] == 1

    def test_num_sorts_third(self):
        assert model_sort_key("num_model")[0] == 2

    def test_approx_sorts_fourth(self):
        assert model_sort_key("approx_model")[0] == 3

    def test_range_sorts_fifth(self):
        assert model_sort_key("range_model")[0] == 4

    def test_other_sorts_last(self):
        assert model_sort_key("some_other_model")[0] == 5

    def test_case_insensitive(self):
        assert model_sort_key("BASELINE")[0] == model_sort_key("baseline")[0]

    def test_alphabetical_tiebreak(self):
        key_a = model_sort_key("alpha")
        key_b = model_sort_key("beta")
        assert key_a < key_b


# ---------------------------------------------------------------------------
# safe_float / safe_int / safe_float_or_nan
# ---------------------------------------------------------------------------


class TestSafeConversions:
    def test_safe_float_numeric_string(self):
        assert safe_float("0.5") == pytest.approx(0.5)

    def test_safe_float_none_returns_none(self):
        assert safe_float(None) is None

    def test_safe_float_invalid_returns_none(self):
        assert safe_float("not_a_number") is None

    def test_safe_int_numeric(self):
        assert safe_int(42) == 42

    def test_safe_int_float_truncates(self):
        assert safe_int(3.9) == 3

    def test_safe_int_none_returns_none(self):
        assert safe_int(None) is None

    def test_safe_int_invalid_returns_none(self):
        assert safe_int("abc") is None

    def test_safe_float_or_nan_valid(self):
        assert safe_float_or_nan(0.9) == pytest.approx(0.9)

    def test_safe_float_or_nan_none_returns_nan(self):
        assert math.isnan(safe_float_or_nan(None))

    def test_safe_float_or_nan_invalid_returns_nan(self):
        assert math.isnan(safe_float_or_nan("bad"))


# ---------------------------------------------------------------------------
# infer_model_name
# ---------------------------------------------------------------------------


class TestInferModelName:
    def test_parent_strategy(self, tmp_path):
        path = tmp_path / "my_model" / "metrics.json"
        assert infer_model_name(path, "parent") == "my_model"

    def test_stem_strategy(self, tmp_path):
        path = tmp_path / "my_model" / "results.json"
        assert infer_model_name(path, "stem") == "results"


# ---------------------------------------------------------------------------
# discover_metrics_files
# ---------------------------------------------------------------------------


class TestDiscoverMetricsFiles:
    def test_finds_files(self, tmp_path):
        (tmp_path / "modelA").mkdir()
        (tmp_path / "modelA" / "metrics.json").write_text("{}")
        (tmp_path / "modelB").mkdir()
        (tmp_path / "modelB" / "metrics.json").write_text("{}")

        found = discover_metrics_files(tmp_path, "**/metrics.json")
        assert len(found) == 2

    def test_no_match_returns_empty(self, tmp_path):
        found = discover_metrics_files(tmp_path, "**/metrics.json")
        assert found == []

    def test_returns_sorted(self, tmp_path):
        for name in ["zmodel", "amodel"]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "metrics.json").write_text("{}")

        found = discover_metrics_files(tmp_path, "**/metrics.json")
        names = [p.parent.name for p in found]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# extract_split_groups
# ---------------------------------------------------------------------------


class TestExtractSplitGroups:
    def test_extracts_true_groups(self):
        data = minimal_metrics(true_group=True, false_group=True)
        result = extract_split_groups(data, "True")
        assert len(result) == 1
        key = list(result.keys())[0]
        assert "_True.json" in key

    def test_extracts_false_groups(self):
        data = minimal_metrics(true_group=True, false_group=True)
        result = extract_split_groups(data, "False")
        assert len(result) == 1
        key = list(result.keys())[0]
        assert "_False.json" in key

    def test_empty_when_no_match(self):
        data = minimal_metrics(true_group=False, false_group=False)
        assert extract_split_groups(data, "True") == {}

    def test_extracted_fields(self):
        data = minimal_metrics(true_group=True, false_group=False)
        groups = extract_split_groups(data, "True")
        entry = next(iter(groups.values()))
        assert entry["accuracy"] == pytest.approx(0.8)
        assert entry["parse_failures"] == 2
        assert entry["total"] == 50
        assert entry["valid"] == 48
        assert entry["correct"] == 40


# ---------------------------------------------------------------------------
# extract_normalized_baseline
# ---------------------------------------------------------------------------


class TestExtractNormalizedBaseline:
    def test_returns_none_when_absent(self):
        data = minimal_metrics(normalized=False)
        assert extract_normalized_baseline(data) is None

    def test_returns_values_when_present(self):
        data = minimal_metrics(normalized=True)
        result = extract_normalized_baseline(data)
        assert result is not None
        assert result["accuracy"] == pytest.approx(0.75)
        assert result["parse_failures"] == 3
        assert result["total"] == 80


# ---------------------------------------------------------------------------
# split_groups_by_perturb_family
# ---------------------------------------------------------------------------


class TestSplitGroupsByPerturbFamily:
    def _make_rows(self, group_names):
        return [
            {"group_name": g, "model": "m", "accuracy": 0.5}
            for g in group_names
        ]

    def test_type1_goes_to_type1(self):
        rows = self._make_rows(["foo | perturb_type1"])
        t1, t23 = split_groups_by_perturb_family(rows)
        assert len(t1) == 1
        assert len(t23) == 0

    def test_type2_goes_to_type23(self):
        rows = self._make_rows(["foo | perturb_type2"])
        t1, t23 = split_groups_by_perturb_family(rows)
        assert len(t1) == 0
        assert len(t23) == 1

    def test_type3_goes_to_type23(self):
        rows = self._make_rows(["foo | perturb_type3"])
        t1, t23 = split_groups_by_perturb_family(rows)
        assert len(t23) == 1

    def test_unrecognized_group_excluded(self):
        rows = self._make_rows(["foo | perturb_typeX"])
        t1, t23 = split_groups_by_perturb_family(rows)
        assert len(t1) == 0
        assert len(t23) == 0

    def test_mixed_groups(self):
        rows = self._make_rows(
            ["a | perturb_type1", "b | perturb_type2", "c | perturb_type3"]
        )
        t1, t23 = split_groups_by_perturb_family(rows)
        assert len(t1) == 1
        assert len(t23) == 2


# ---------------------------------------------------------------------------
# to_wide
# ---------------------------------------------------------------------------


class TestToWide:
    def _make_rows(self):
        return [
            {"group_name": "groupA", "model": "baseline", "accuracy": 0.9},
            {"group_name": "groupA", "model": "num_model", "accuracy": 0.7},
            {"group_name": "groupB", "model": "baseline", "accuracy": 0.8},
        ]

    def test_column_headers(self):
        rows = self._make_rows()
        fields, _ = to_wide(rows, "accuracy")
        assert fields[0] == "group_name"
        assert "baseline" in fields
        assert "num_model" in fields

    def test_model_sort_order(self):
        rows = self._make_rows()
        fields, _ = to_wide(rows, "accuracy")
        model_cols = fields[1:]
        assert model_cols == sorted(model_cols, key=model_sort_key)

    def test_pivot_values(self):
        rows = self._make_rows()
        _, out_rows = to_wide(rows, "accuracy")
        group_a = next(r for r in out_rows if r["group_name"] == "groupA")
        assert group_a["baseline"] == pytest.approx(0.9)
        assert group_a["num_model"] == pytest.approx(0.7)

    def test_missing_model_value_is_none(self):
        rows = self._make_rows()
        _, out_rows = to_wide(rows, "accuracy")
        group_b = next(r for r in out_rows if r["group_name"] == "groupB")
        assert group_b.get("num_model") is None


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_creates_file(self, tmp_path):
        rows = [{"a": 1, "b": 2}]
        out = tmp_path / "out.csv"
        write_csv(out, rows, ["a", "b"])
        assert out.exists()

    def test_header_and_row(self, tmp_path):
        rows = [{"x": "hello", "y": 42}]
        out = tmp_path / "out.csv"
        write_csv(out, rows, ["x", "y"])
        with out.open(encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert reader[0]["x"] == "hello"
        assert reader[0]["y"] == "42"

    def test_creates_parent_directories(self, tmp_path):
        rows = [{"a": 1}]
        out = tmp_path / "nested" / "dir" / "out.csv"
        write_csv(out, rows, ["a"])
        assert out.exists()


# ---------------------------------------------------------------------------
# build_rows (integration)
# ---------------------------------------------------------------------------


class TestBuildRows:
    def test_overall_rows_populated(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics())
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        _, overall_rows, _ = build_rows(files, tmp_path, "parent", "True")
        assert len(overall_rows) == 1
        assert overall_rows[0]["model"] == "model_a"
        assert overall_rows[0]["overall_total"] == 100
        assert overall_rows[0]["overall_parse_failures"] == 5

    def test_overall_parse_failure_rate(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics())
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        _, overall_rows, _ = build_rows(files, tmp_path, "parent", "True")
        rate = overall_rows[0]["overall_parse_failure_rate"]
        assert rate == pytest.approx(5 / 100)

    def test_split_rows_for_true(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics())
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        split_rows, _, _ = build_rows(files, tmp_path, "parent", "True")
        assert len(split_rows) == 1
        assert "_True.json" in split_rows[0]["group_name"]

    def test_split_rows_for_false(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics())
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        split_rows, _, _ = build_rows(files, tmp_path, "parent", "False")
        assert len(split_rows) == 1
        assert "_False.json" in split_rows[0]["group_name"]

    def test_normalized_rows_present_when_in_data(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics(normalized=True))
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        _, _, normalized_rows = build_rows(files, tmp_path, "parent", "True")
        assert len(normalized_rows) == 1
        assert normalized_rows[0]["accuracy"] == pytest.approx(0.75)

    def test_normalized_rows_empty_when_absent(self, tmp_path):
        make_metrics_file(
            tmp_path, "model_a", minimal_metrics(normalized=False)
        )
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        _, _, normalized_rows = build_rows(files, tmp_path, "parent", "True")
        assert normalized_rows == []

    def test_multiple_models(self, tmp_path):
        make_metrics_file(tmp_path, "model_a", minimal_metrics())
        make_metrics_file(tmp_path, "model_b", minimal_metrics())
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        split_rows, overall_rows, _ = build_rows(
            files, tmp_path, "parent", "True"
        )
        model_names = {r["model"] for r in overall_rows}
        assert model_names == {"model_a", "model_b"}
        assert len(split_rows) == 2

    def test_stem_strategy(self, tmp_path):
        path = make_metrics_file(tmp_path, "model_a", minimal_metrics())
        _, overall_rows, _ = build_rows([path], tmp_path, "stem", "True")
        assert overall_rows[0]["model"] == "metrics"

    def test_parse_failure_rate_none_when_total_zero(self, tmp_path):
        data = minimal_metrics(overall_total=0, overall_parse_failures=0)
        make_metrics_file(tmp_path, "model_a", data)
        files = discover_metrics_files(tmp_path, "**/metrics.json")
        _, overall_rows, _ = build_rows(files, tmp_path, "parent", "True")
        assert overall_rows[0]["overall_parse_failure_rate"] is None
