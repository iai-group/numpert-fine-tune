"""Demo script to show the before/after output of the LaTeX table fix."""

from src.metrics.eval_metrics_latex import DeltaEvalTable

# Create test data
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

print("Generated LaTeX output:")
print("=" * 80)
print(latex)
print("=" * 80)
print("\nKey checks:")
section_a = "\\textbf{Section A}"
section_b = "\\textbf{Section B}"
print(f"✓ Section A present: {section_a in latex}")
print(f"✓ Section B present: {section_b in latex}")
print(f"✓ model1 present: {'model1' in latex}")
print(f"✓ model2 present: {'model2' in latex}")
