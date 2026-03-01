"""Tests for the metrics module."""

import json
import os
from typing import Dict, List, Optional

from src.llm_finetune.utils.inference.metrics import (
    ClassificationScores,
    MetricsReport,
    build_result_records,
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
    save_parse_failures,
    save_predictions_csv,
    _compute_scores,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

LABEL_TRUE = '{"label": true}'
LABEL_FALSE = '{"label": false}'


def _make_results(
    labels: List[str],
    predictions: List[Optional[str]],
    perturbation_types: Optional[List[str]] = None,
    source_files: Optional[List[str]] = None,
) -> List[Dict]:
    """Build a list of result dicts for testing."""
    n = len(labels)
    if perturbation_types is None:
        perturbation_types = ["original_normalized"] * n
    if source_files is None:
        source_files = ["test_file.json"] * n

    results = []
    for i in range(n):
        record: Dict = {
            "claim_id": f"claim_{i}",
            "label": labels[i],
            "prediction": predictions[i],
            "perturbation_type": perturbation_types[i],
            "source_file": source_files[i],
        }
        results.append(record)
    return results


# ---------------------------------------------------------------------------
# _compute_scores
# ---------------------------------------------------------------------------


class TestComputeScores:
    """Tests for the _compute_scores helper."""

    def test_perfect_predictions(self):
        """All predictions correct → accuracy 1.0."""
        true = [LABEL_TRUE, LABEL_FALSE, LABEL_TRUE]
        pred = [LABEL_TRUE, LABEL_FALSE, LABEL_TRUE]
        scores = _compute_scores("test_group", true, pred)

        assert scores.group_name == "test_group"
        assert scores.accuracy == 1.0
        assert scores.f1_macro == 1.0
        assert scores.total == 3
        assert scores.correct == 3
        assert scores.parse_failures == 0

    def test_all_wrong_predictions(self):
        """All predictions wrong → accuracy 0.0."""
        true = [LABEL_TRUE, LABEL_FALSE]
        pred = [LABEL_FALSE, LABEL_TRUE]
        scores = _compute_scores("wrong", true, pred)

        assert scores.accuracy == 0.0
        assert scores.correct == 0

    def test_parse_failures_excluded(self):
        """None predictions should be counted as failures, not errors."""
        true = [LABEL_TRUE, LABEL_FALSE, LABEL_TRUE, LABEL_FALSE]
        pred = [LABEL_TRUE, None, LABEL_TRUE, None]
        scores = _compute_scores("with_failures", true, pred)

        assert scores.total == 4
        assert scores.parse_failures == 2
        assert scores.valid == 2
        assert scores.accuracy == 1.0  # 2/2 valid are correct
        assert scores.correct == 2

    def test_all_failures(self):
        """When every prediction is None, metrics should still be returned."""
        true = [LABEL_TRUE, LABEL_FALSE]
        pred = [None, None]
        scores = _compute_scores("all_fail", true, pred)

        assert scores.total == 2
        assert scores.parse_failures == 2
        assert scores.valid == 0
        assert scores.accuracy == 0.0

    def test_empty_inputs(self):
        """Empty lists should produce zero scores."""
        scores = _compute_scores("empty", [], [])
        assert scores.total == 0
        assert scores.accuracy == 0.0

    def test_classification_report_populated(self):
        """The report string should be non-empty when valid."""
        true = [LABEL_TRUE, LABEL_FALSE]
        pred = [LABEL_TRUE, LABEL_FALSE]
        scores = _compute_scores("report", true, pred)
        assert len(scores.classification_report_str) > 0

    def test_precision_and_recall(self):
        """Precision and recall should be computed."""
        true = [LABEL_TRUE, LABEL_TRUE, LABEL_FALSE]
        pred = [LABEL_TRUE, LABEL_FALSE, LABEL_FALSE]
        scores = _compute_scores("prec_rec", true, pred)

        assert scores.precision_macro > 0
        assert scores.recall_macro > 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the top-level compute_metrics function."""

    def test_overall_metrics(self):
        """Overall metrics should always be computed."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE],
            predictions=[LABEL_TRUE, LABEL_FALSE, LABEL_FALSE],
        )
        report = compute_metrics(results)

        assert report.overall is not None
        assert report.overall.total == 3
        assert report.overall.correct == 2

    def test_per_perturbation_type(self):
        """Should separate metrics by perturbation type."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, LABEL_FALSE, LABEL_FALSE, LABEL_FALSE],
            perturbation_types=[
                "original_normalized",
                "original_normalized",
                "perturb_type1",
                "perturb_type1",
            ],
        )
        report = compute_metrics(results)

        assert "original_normalized" in report.per_perturbation_type
        assert "perturb_type1" in report.per_perturbation_type

        original = report.per_perturbation_type["original_normalized"]
        assert original.total == 2
        assert original.accuracy == 1.0

        type1 = report.per_perturbation_type["perturb_type1"]
        assert type1.total == 2
        assert type1.correct == 1

    def test_per_source_file(self):
        """Should separate metrics by source file."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE],
            predictions=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE],
            source_files=[
                "approximation_True.json",
                "approximation_True.json",
                "numeration_False.json",
            ],
        )
        report = compute_metrics(results)

        assert "approximation_True.json" in report.per_source_file
        assert "numeration_False.json" in report.per_source_file
        assert report.per_source_file["approximation_True.json"].total == 2
        assert report.per_source_file["numeration_False.json"].total == 1

    def test_per_file_and_type_combination(self):
        """Should separate metrics by file + perturbation type combo."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE, LABEL_FALSE],
            perturbation_types=[
                "original_normalized",
                "perturb_type1",
                "original_normalized",
                "perturb_type2",
            ],
            source_files=[
                "approx_True.json",
                "approx_True.json",
                "numer_False.json",
                "numer_False.json",
            ],
        )
        report = compute_metrics(results)

        assert len(report.per_file_and_type) == 4
        expected_keys = {
            "approx_True.json | original_normalized",
            "approx_True.json | perturb_type1",
            "numer_False.json | original_normalized",
            "numer_False.json | perturb_type2",
        }
        assert set(report.per_file_and_type.keys()) == expected_keys

    def test_no_source_file_skips_file_breakdowns(self):
        """When source_file is absent, per-file sections should be empty."""
        results = [
            {
                "claim_id": "c1",
                "label": LABEL_TRUE,
                "prediction": LABEL_TRUE,
                "perturbation_type": "original_normalized",
            }
        ]
        report = compute_metrics(results)
        assert len(report.per_source_file) == 0
        assert len(report.per_file_and_type) == 0

    def test_disable_breakdowns(self):
        """Flags should control which breakdowns are computed."""
        results = _make_results(
            labels=[LABEL_TRUE],
            predictions=[LABEL_TRUE],
        )
        report = compute_metrics(
            results,
            include_per_type=False,
            include_per_file=False,
            include_per_file_and_type=False,
        )
        assert len(report.per_perturbation_type) == 0
        assert len(report.per_source_file) == 0
        assert len(report.per_file_and_type) == 0
        assert report.overall is not None

    def test_parse_failures_per_type(self):
        """Parse failures should be tracked per perturbation type."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, None, None],
            perturbation_types=[
                "original_normalized",
                "perturb_type1",
                "perturb_type1",
            ],
        )
        report = compute_metrics(results)

        original = report.per_perturbation_type["original_normalized"]
        assert original.parse_failures == 0

        type1 = report.per_perturbation_type["perturb_type1"]
        assert type1.parse_failures == 2
        assert type1.valid == 0


# ---------------------------------------------------------------------------
# MetricsReport.to_dict / ClassificationScores.to_dict
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for to_dict methods."""

    def test_classification_scores_to_dict(self):
        """to_dict should return all expected keys."""
        scores = ClassificationScores(
            group_name="test",
            accuracy=0.95,
            total=100,
            correct=95,
            parse_failures=5,
        )
        d = scores.to_dict()
        assert d["group_name"] == "test"
        assert d["accuracy"] == 0.95
        assert d["total"] == 100
        assert d["valid"] == 95
        assert d["parse_failures"] == 5

    def test_metrics_report_to_dict_structure(self):
        """Full report dict should have the expected top-level keys."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, LABEL_FALSE],
            perturbation_types=["original_normalized", "perturb_type1"],
            source_files=["file_a.json", "file_b.json"],
        )
        report = compute_metrics(results)
        d = report.to_dict()

        assert "overall" in d
        assert "per_perturbation_type" in d
        assert "per_source_file" in d
        assert "per_file_and_type" in d

    def test_to_dict_is_json_serializable(self):
        """to_dict output must be serializable to JSON."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, None],
        )
        report = compute_metrics(results)
        # Should not raise
        json.dumps(report.to_dict())


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestIO:
    """Tests for save functions."""

    def test_save_metrics_report(self, tmp_path):
        """Should write metrics.json with expected structure."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, LABEL_FALSE],
        )
        report = compute_metrics(results)
        path = save_metrics_report(report, str(tmp_path))

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "overall" in data
        assert data["overall"]["accuracy"] == 1.0

    def test_save_parse_failures_with_failures(self, tmp_path):
        """Should create file when there are failures."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, None],
        )
        raw_outputs = [
            {"index": 0, "raw_response": '{"label": true}'},
            {"index": 1, "raw_response": "garbage"},
        ]
        path = save_parse_failures(results, raw_outputs, str(tmp_path))

        assert path is not None
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["raw_response"] == "garbage"

    def test_save_parse_failures_without_failures(self, tmp_path):
        """Should return None when there are no failures."""
        results = _make_results(
            labels=[LABEL_TRUE],
            predictions=[LABEL_TRUE],
        )
        raw_outputs = [
            {"index": 0, "raw_response": '{"label": true}'},
        ]
        path = save_parse_failures(results, raw_outputs, str(tmp_path))
        assert path is None

    def test_save_predictions_csv(self, tmp_path):
        """Should produce a CSV with expected columns."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE],
            predictions=[LABEL_TRUE, None],
            source_files=["a.json", "b.json"],
        )
        raw_outputs = [
            {"raw_response": '{"label": true}'},
            {"raw_response": "??"},
        ]
        path = save_predictions_csv(results, raw_outputs, str(tmp_path))

        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 2
        assert "source_file" in df.columns
        assert "perturbation_type" in df.columns
        assert df.iloc[1]["prediction"] == "PARSE_FAILURE"


# ---------------------------------------------------------------------------
# build_result_records
# ---------------------------------------------------------------------------


class TestBuildResultRecords:
    """Tests for build_result_records."""

    def test_basic(self):
        """Should build records with expected fields."""
        from datasets import Dataset

        test_data = Dataset.from_list(
            [
                {
                    "claim_id": "c1",
                    "label": LABEL_TRUE,
                    "perturbation_type": "original_normalized",
                    "source_file": "approx.json",
                    "conversations": [],
                },
                {
                    "claim_id": "c2",
                    "label": LABEL_FALSE,
                    "perturbation_type": "perturb_type1",
                    "source_file": "approx.json",
                    "conversations": [],
                },
            ]
        )
        raw_outputs = [
            {"raw_response": '{"label": true}'},
            {"raw_response": "bad"},
        ]
        predictions = [LABEL_TRUE, None]

        records = build_result_records(test_data, raw_outputs, predictions)

        assert len(records) == 2
        assert records[0]["prediction"] == LABEL_TRUE
        assert records[0]["source_file"] == "approx.json"
        assert records[1]["prediction"] is None

    def test_no_source_file_column(self):
        """Should work without source_file column."""
        from datasets import Dataset

        test_data = Dataset.from_list(
            [
                {
                    "claim_id": "c1",
                    "label": LABEL_TRUE,
                    "perturbation_type": "original_normalized",
                    "conversations": [],
                },
            ]
        )
        raw_outputs = [{"raw_response": '{"label": true}'}]
        predictions = [LABEL_TRUE]

        records = build_result_records(test_data, raw_outputs, predictions)
        assert "source_file" not in records[0]


# ---------------------------------------------------------------------------
# print_metrics_report (smoke test)
# ---------------------------------------------------------------------------


class TestPrintMetricsReport:
    """Smoke tests for the print function — should not raise."""

    def test_prints_without_error(self, capsys):
        """Full report should print cleanly."""
        results = _make_results(
            labels=[LABEL_TRUE, LABEL_FALSE, LABEL_TRUE],
            predictions=[LABEL_TRUE, LABEL_FALSE, None],
            perturbation_types=[
                "original_normalized",
                "perturb_type1",
                "perturb_type2",
            ],
            source_files=["a.json", "a.json", "b.json"],
        )
        report = compute_metrics(results)
        print_metrics_report(report, show_classification_reports=True)

        captured = capsys.readouterr()
        assert "OVERALL" in captured.out
        assert "PERTURBATION TYPE" in captured.out
        assert "SOURCE FILE" in captured.out

    def test_prints_empty_report(self, capsys):
        """Empty report should print without crashing."""
        report = MetricsReport()
        print_metrics_report(report)
        # Should not raise
