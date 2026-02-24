"""Evaluation table with baseline + deltas LaTeX rendering."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass
class DeltaEvalTable:
    """Collects evaluation results and renders baseline+deltas LaTeX table."""

    metrics: Tuple[str, ...] = (
        "accuracy",
    )  # keep one main metric (usually accuracy)
    average: str = "auto"  # for PR/F1
    digits: int = 2
    higher_is_better: bool = True
    experiment_type: str = ""

    # storage: model -> variation -> (y_true, y_pred)
    _store: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )

    # ------------------------- Data ingestion -------------------------
    def add_result(
        self,
        model: str,
        variation: str,
        y_true: List[Any],
        y_pred: List[Any],
    ) -> None:
        """Add one run for a model and variation.

        Args:
            model: Model name.
            variation: Variation name (e.g. "original", "approx").
            y_true: List of true labels.
            y_pred: List of predicted labels.

        Raises:
            ValueError: If y_true and y_pred have different lengths.
        """
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if yt.shape != yp.shape:
            raise ValueError("y_true and y_pred must have same shape")
        self._store.setdefault(model, {})[variation] = (yt, yp)

    # ------------------------- Metric calculation -------------------------
    def _infer_average(self, y_true, y_pred) -> str:
        if self.average != "auto":
            return self.average
        labels = np.unique(np.concatenate([y_true, y_pred]))
        return "macro" if len(labels) > 2 else "binary"

    def _compute_metric(self, metric: str, y_true, y_pred) -> float:
        avg = self._infer_average(y_true, y_pred)
        if metric == "accuracy":
            return accuracy_score(y_true, y_pred) * 100
        elif metric == "precision":
            return (
                precision_score(y_true, y_pred, average=avg, zero_division=0)
                * 100
            )
        elif metric == "recall":
            return (
                recall_score(y_true, y_pred, average=avg, zero_division=0) * 100
            )
        elif metric == "f1":
            return f1_score(y_true, y_pred, average=avg, zero_division=0) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_scores(self) -> Dict[str, Dict[str, float]]:
        """Return computed metric values per model/variation."""
        scores: Dict[str, Dict[str, float]] = {}
        metric = self.metrics[0]
        for model, runs in self._store.items():
            scores[model] = {}
            for variation, (yt, yp) in runs.items():
                if len(yt) == 0 or len(yp) == 0:
                    continue
                val = self._compute_metric(metric, yt, yp)
                scores[model][variation] = val
        return scores

    # ------------------------- Formatting helpers --------------------
    def _fmt_score(self, val: float) -> str:
        return f"{val:.{self.digits}f}"

    def _fmt_delta(self, base: float, val: float) -> str:
        delta = val - base
        if abs(delta) < 1e-6:
            return ""
        color = "darkgreen" if (delta > 0) == self.higher_is_better else "red"
        sign = "+" if delta > 0 else ""
        return f"\\textsuperscript{{\\textcolor{{{color}}}{{{sign}{delta:.{self.digits}f}}}}}"  # noqa: E501

    # ------------------------- LaTeX rendering -----------------------
    def emit_row(
        self,
        model: str,
        best_scores: Dict[str, float],
        best_original: float,
        scores: Dict[str, Dict[str, float]],
        variations: List[str],
    ) -> List[str]:
        """Emit one row for a model with deltas from original.

        Args:
            model: Model name.
            best_scores: Dict of best scores for each variation.
            best_original: Best score for the original variation.
            scores: Full scores dict.
            variations: List of variation names (order), excluding 'original'.

        Returns:
            List of row cells for LaTeX table.
        """
        sc = scores[model]
        if "original" not in sc:
            raise ValueError(f"Model {model} missing 'original' variation")
        base = sc["original"]
        formatted_base = self._fmt_score(base)
        if abs(base - best_original) < 1e-6:
            formatted_base = f"\\textbf{{{formatted_base}}}"
        model_name = model.split("_")[0]  # remove section suffix if any
        row_cells = [model_name, formatted_base]
        for v in variations:
            if v in sc:
                val = sc[v]
                formatted_val = self._fmt_score(val)
                if abs(val - best_scores[v]) < 1e-6:
                    formatted_val = f"\\textbf{{{formatted_val}}}"
                cell = formatted_val + self._fmt_delta(base, val)
            else:
                cell = "--"
            row_cells.append(cell)
        return row_cells

    def to_latex(
        self,
        variations: List[str],
        section_labels: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Render LaTeX table with bolded best values in each column.

        Args:
            variations: List of variation names (order), excluding 'original'.
            section_labels: Optional dict of section_title -> list of models.
        """
        scores = self.get_scores()

        # Sort models alphabetically
        sorted_models = sorted(scores.keys())

        # header
        header = ["\\textbf{Model}", "\\textbf{Original}"] + [
            f"\\textbf{{{v.capitalize()}}}" for v in variations
        ]
        lines = []
        lines.append("\\begin{table*}[ht]")
        lines.append("\\begin{tabular}{l" + "c" * (len(variations) + 1) + "}")
        lines.append("\\hline")
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        # Find best scores for each variation and original
        best_scores = {}
        for v in variations:
            best_scores[v] = max(
                (scores[model][v] for model in scores if v in scores[model]),
                default=float("-inf"),
            )
        best_original = max(
            (
                scores[model]["original"]
                for model in scores
                if "original" in scores[model]
            ),
            default=float("-inf"),
        )

        if section_labels:
            for section, models in section_labels.items():
                lines.append(
                    "\\multicolumn{"
                    + str(len(variations) + 2)
                    + "}{c}{\\textbf{"
                    + section
                    + "}}  \\\\"
                )
                lines.append("\\midrule")
                for m in sorted(models):
                    if m in scores:
                        row_cells = self.emit_row(
                            m,
                            best_scores,
                            best_original,
                            scores,
                            variations,
                        )
                        lines.append(" & ".join(row_cells) + " \\\\")
                lines.append("\\midrule")
        else:
            for m in sorted_models:
                row_cells = self.emit_row(
                    m, best_scores, best_original, scores, variations
                )
                lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        return "\n".join(lines)


# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Example: binary classification
    y_true = list(rng.integers(0, 2, size=100))
    y_pred_good = list(y_true.copy())
    y_pred_good[
        rng.choice(len(y_true), 10, replace=False)
    ] ^= 1  # flip 10 labels
    y_pred_good = list(y_pred_good)
    y_pred_bad = list(rng.integers(0, 2, size=100))

    table = DeltaEvalTable(metrics=("accuracy",), digits=2)

    table.add_result("Model A", "original", y_true, y_pred_good)
    table.add_result("Model A", "approx", y_true, y_pred_bad)
    table.add_result("Model A", "range", y_true, y_pred_good)
    table.add_result("Model A", "random", y_true, y_pred_good)
    table.add_result("Model A", "numeration", y_true, y_pred_good)
    table.add_result("Model A", "negation", y_true, y_pred_good)
    table.add_result("Model A", "masking", y_true, y_pred_good)

    table.add_result("Model B", "original", y_true, y_pred_bad)
    table.add_result("Model B", "approx", y_true, y_pred_good)
    table.add_result("Model B", "range", y_true, y_pred_bad)
    table.add_result("Model B", "random", y_true, y_pred_good)
    table.add_result("Model B", "numeration", y_true, y_pred_good)
    table.add_result("Model B", "negation", y_true, y_pred_good)
    table.add_result("Model B", "masking", y_true, y_pred_good)

    latex = table.to_latex(
        variations=[
            "approx",
            "range",
            "random",
            "numeration",
            "negation",
            "masking",
        ],
        section_labels={"Test Models": ["Model A", "Model B"]},
    )
    print(latex)
