# Suggested Run Commands

## Full run (True + False splits, grouped heatmaps, normalized charts, abbreviated names)

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false \
  --charts \
  --include-normalized-in-charts \
  --abbreviate-model-names
```

## Full run without name abbreviation

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

## CSV-only run with abbreviated model names

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false \
  --abbreviate-model-names
```

## Custom metrics-file pattern (e.g. only a specific subdirectory)

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference \
  --output-dir results/inference/comparison \
  --metrics-pattern "qwen3-8b*/metrics.json" \
  --charts \
  --abbreviate-model-names
```

## Use filename stem instead of parent folder as model name

```bash
python -m src.llm_finetune.model_compare.compare_metrics \
  --input-dir results/inference/qwen3-8b \
  --output-dir results/inference/qwen3-8b/model_comparison_true_false \
  --model-name-from stem \
  --charts
```

## CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Root directory scanned recursively for `metrics.json` files |
| `--output-dir` | *(required)* | Directory where CSV files and optional charts are written |
| `--metrics-pattern` | `**/metrics.json` | Glob pattern (relative to `--input-dir`) used to locate metrics files |
| `--charts` | off | Generate PNG heatmaps and bar charts under `<output-dir>/<split>/charts/` |
| `--include-normalized-in-charts` | off | Include `original_normalized` baseline in chart outputs (requires `--charts`) |
| `--model-name-from` | `parent` | How model names are inferred: `parent` = parent folder name, `stem` = filename stem |
| `--abbreviate-model-names` | off | Shorten names to `<family>-<size>-<tag>` (e.g. `qwen3-8b-baseline`). Applied to both CSV outputs and chart labels. |

## Output locations

- Root `--output-dir` contains:
  - `overall_parse_failures_summary.csv`
  - `normalized_baseline_metrics.csv` *(if any model has `original_normalized` data)*
- `true_split/` and `false_split/` subdirs each contain:
  - `perturb_type1_metrics_long.csv`
  - `perturb_type1_accuracy_wide.csv`
  - `perturb_type1_parse_failures_wide.csv`
  - `perturb_type23_metrics_long.csv`
  - `perturb_type23_accuracy_wide.csv`
  - `perturb_type23_parse_failures_wide.csv`
  - `charts/` *(if `--charts` was set)*
