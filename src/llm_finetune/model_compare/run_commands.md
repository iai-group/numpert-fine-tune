# Suggested Run Commands

## Full run (True + False splits, grouped heatmaps, normalized charts)

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false \
  --charts \
  --include-normalized-in-charts
```

## CSV-only run (no charts)

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false
```

## Run with explicit venv Python

```bash
  python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false \
  --charts \
  --include-normalized-in-charts
```

## Output locations

- Root output dir contains:
  - overall_parse_failures_summary.csv
  - normalized_baseline_metrics.csv
- Split subdirs contain split-specific CSV and chart outputs:
  - true_split/
  - false_split/
