"""Tests for the DeltaEvalTable class in the eval_metrics_latex module."""

import pytest
import numpy as np
from src.metrics.eval_metrics_latex import DeltaEvalTable


def test_init_default_values():
    """Test initialization with default values."""
    table = DeltaEvalTable()
    assert table.metrics == ("accuracy",)
    assert table.average == "auto"
    assert table.digits == 2
    assert table.higher_is_better is True
    assert table._store == {}


def test_init_custom_values():
    """Test initialization with custom values."""
    table = DeltaEvalTable(
        metrics=("precision", "recall"),
        average="macro",
        digits=3,
        higher_is_better=False,
    )
    assert table.metrics == ("precision", "recall")
    assert table.average == "macro"
    assert table.digits == 3
    assert table.higher_is_better is False


def test_add_result():
    """Test adding results to the table."""
    table = DeltaEvalTable()
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    table.add_result("model1", "original", y_true, y_pred)

    assert "model1" in table._store
    assert "original" in table._store["model1"]
    np.testing.assert_array_equal(
        table._store["model1"]["original"][0], np.array(y_true)
    )
    np.testing.assert_array_equal(
        table._store["model1"]["original"][1], np.array(y_pred)
    )


def test_add_result_different_shapes():
    """Test adding results with different shapes raises error."""
    table = DeltaEvalTable()
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1]

    with pytest.raises(ValueError):
        table.add_result("model1", "original", y_true, y_pred)


def test_infer_average_binary():
    """Test auto-inferred average for binary classification."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    assert table._infer_average(y_true, y_pred) == "binary"


def test_infer_average_multi():
    """Test auto-inferred average for multi-class classification."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 2])

    assert table._infer_average(y_true, y_pred) == "macro"


def test_infer_average_custom():
    """Test custom average setting is respected."""
    table = DeltaEvalTable(average="weighted")
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    assert table._infer_average(y_true, y_pred) == "weighted"


def test_compute_metric_accuracy():
    """Test accuracy metric computation."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # 3 out of 4 correct = 75%
    assert table._compute_metric("accuracy", y_true, y_pred) == 75.0


def test_compute_metric_precision():
    """Test precision metric computation."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # For binary with class 1: 2 true positives, 1 false positive = 2/3 ≈ 66.67%
    assert abs(table._compute_metric("precision", y_true, y_pred) - 66.67) < 0.1


def test_compute_metric_recall():
    """Test recall metric computation."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # For class 1: 2 true positives, 0 false negatives = 2/2 = 100%
    assert table._compute_metric("recall", y_true, y_pred) == 100.0


def test_compute_metric_f1():
    """Test F1 score computation."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # F1 should be between precision and recall
    f1 = table._compute_metric("f1", y_true, y_pred)
    precision = table._compute_metric("precision", y_true, y_pred)
    recall = table._compute_metric("recall", y_true, y_pred)

    assert min(precision, recall) <= f1 <= max(precision, recall)


def test_compute_metric_unknown():
    """Test unknown metric raises ValueError."""
    table = DeltaEvalTable()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    with pytest.raises(ValueError):
        table._compute_metric("unknown_metric", y_true, y_pred)


def test_get_scores():
    """Test retrieving computed scores for models and variations."""
    table = DeltaEvalTable()
    y_true = [0, 1, 0, 1]
    y_pred_good = [0, 1, 0, 1]  # 100% accuracy
    y_pred_bad = [1, 0, 1, 0]  # 0% accuracy

    table.add_result("model1", "original", y_true, y_pred_good)
    table.add_result("model1", "variation", y_true, y_pred_bad)
    table.add_result("model2", "original", y_true, y_pred_bad)

    scores = table.get_scores()

    assert "model1" in scores
    assert "model2" in scores
    assert "original" in scores["model1"]
    assert "variation" in scores["model1"]
    assert "original" in scores["model2"]
    assert scores["model1"]["original"] == 100.0
    assert scores["model1"]["variation"] == 0.0
    assert scores["model2"]["original"] == 0.0


def test_fmt_score():
    """Test score formatting with different precision settings."""
    table = DeltaEvalTable(digits=2)
    assert table._fmt_score(75.123) == "75.12"

    table = DeltaEvalTable(digits=3)
    assert table._fmt_score(75.123) == "75.123"


def test_fmt_delta():
    """Test delta formatting in different scenarios."""
    # Positive delta, higher is better
    table = DeltaEvalTable(digits=2, higher_is_better=True)
    delta = table._fmt_delta(75.0, 80.0)
    assert "darkgreen" in delta
    assert "+5.00" in delta

    # Negative delta, higher is better
    delta = table._fmt_delta(75.0, 70.0)
    assert "red" in delta
    assert "-5.00" in delta

    # No change
    delta = table._fmt_delta(75.0, 75.0)
    assert delta == ""

    # Positive delta, lower is better
    table = DeltaEvalTable(digits=2, higher_is_better=False)
    delta = table._fmt_delta(75.0, 80.0)
    assert "red" in delta
    assert "+5.00" in delta

    # Negative delta, lower is better
    delta = table._fmt_delta(75.0, 70.0)
    assert "darkgreen" in delta
    assert "-5.00" in delta


def test_emit_row():
    """Test row emission for LaTeX table."""
    table = DeltaEvalTable()

    scores = {"model1": {"original": 80.0, "var1": 85.0, "var2": 75.0}}

    best_scores = {"var1": 85.0, "var2": 75.0}
    best_original = 90.0  # Some other model has better score

    row = table.emit_row(
        "model1", best_scores, best_original, scores, ["var1", "var2"]
    )

    assert row[0] == "model1"  # Model name
    assert "80.00" in row[1]  # Original score
    assert "\\textbf" not in row[1]  # Not best original
    assert "85.00" in row[2]  # var1 score
    assert "\\textbf" in row[2]  # Best var1
    assert "darkgreen" in row[2]  # Positive delta
    assert "+5.00" in row[2]  # Delta value
    assert "75.00" in row[3]  # var2 score
    assert "\\textbf" in row[3]  # Best var2
    assert "red" in row[3]  # Negative delta
    assert "-5.00" in row[3]  # Delta value


def test_emit_row_missing_original():
    """Test row emission fails when original variation is missing."""
    table = DeltaEvalTable()

    scores = {"model1": {"var1": 85.0, "var2": 75.0}}

    best_scores = {"var1": 85.0, "var2": 75.0}
    best_original = 90.0

    with pytest.raises(ValueError):
        table.emit_row(
            "model1", best_scores, best_original, scores, ["var1", "var2"]
        )


def test_emit_row_missing_variation():
    """Test row emission handles missing variations."""
    table = DeltaEvalTable()

    scores = {
        "model1": {
            "original": 80.0,
            "var1": 85.0,
        }
    }

    best_scores = {"var1": 85.0, "var2": 75.0}
    best_original = 90.0

    row = table.emit_row(
        "model1", best_scores, best_original, scores, ["var1", "var2"]
    )

    assert row[3] == "--"  # Missing var2 should be displayed as --


def test_to_latex_basic():
    """Test basic LaTeX table generation."""
    table = DeltaEvalTable()
    y_true = [0, 1, 0, 1]
    y_pred_good = [0, 1, 0, 1]  # 100% accuracy
    y_pred_bad = [1, 0, 1, 0]  # 0% accuracy

    table.add_result("model1", "original", y_true, y_pred_good)
    table.add_result("model1", "var1", y_true, y_pred_bad)

    latex = table.to_latex(variations=["var1"])

    assert "\\begin{table*}" in latex
    assert "\\end{table*}" in latex
    assert "\\textbf{Model}" in latex
    assert "\\textbf{Original}" in latex
    assert "\\textbf{Var1}" in latex
    assert "model1" in latex
    assert "100.00" in latex  # Original score
    assert "0.00" in latex  # var1 score
    assert "red" in latex  # Negative delta


def test_to_latex_with_sections():
    """Test LaTeX table generation with section labels."""
    table = DeltaEvalTable()
    y_true = [0, 1, 0, 1]
    y_pred_good = [0, 1, 0, 1]  # 100% accuracy
    y_pred_bad = [1, 0, 1, 0]  # 0% accuracy

    table.add_result("model1", "original", y_true, y_pred_good)
    table.add_result("model1", "var1", y_true, y_pred_bad)
    table.add_result("model2", "original", y_true, y_pred_bad)
    table.add_result("model2", "var1", y_true, y_pred_good)

    latex = table.to_latex(
        variations=["var1"],
        section_labels={"Section A": ["model1"], "Section B": ["model2"]},
    )

    assert "\\textbf{Section A}" in latex
    assert "\\textbf{Section B}" in latex
    assert "model1" in latex
    assert "model2" in latex
