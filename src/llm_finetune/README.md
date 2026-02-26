# LLM Fine-Tuning Module

This module implements the full fine-tuning and inference pipeline for numerical perturbation-aware fact-checking models. It uses [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning and supports custom batch sampling strategies that group original and perturbed claims together.

---

## Directory Structure

```
src/llm_finetune/
├── batch_sampler.py               # Perturbation-aware batch sampler
├── config.py                      # Path prefix configuration
├── unsloth_finetune.py            # Training entry point
├── unsloth_inference.py           # Inference entry point
├── upload_model_hf.py             # Upload trained model to HuggingFace Hub
└── data_loading_utils/
    ├── perturbation_types.py      # PerturbationType enum, FileLoadConfig
    ├── perturbation_logic.py      # expand_perturbations() core engine
    ├── conversation.py            # Convert examples to SFTTrainer chat format
    ├── class_balancer.py          # Label balancing & perturbation ratio control
    ├── hf_dataset_instructions.py # High-level data loading API (orchestration)
    └── sequence_len_utility.py    # Sequence length analysis & plotting
```

---

## Data Loading Pipeline

The data loading pipeline converts raw perturbed JSON files into a HuggingFace `Dataset` ready for `SFTTrainer`. The flow is:

```
Raw JSON files
  → load_json_files()                   # Read files into list of dicts
  → expand_perturbations()              # Explode each example into 1 original + N perturbed rows
  → convert_to_conversations()          # Format as system/user/assistant chat messages
  → Dataset.from_list()                 # HuggingFace Dataset
```

### Entry Points

There are two high-level functions in `hf_dataset_instructions.py`:

| Function | Purpose |
|---|---|
| `load_fact_checking_data()` | Load a single split (train/val/test) |
| `load_train_and_validation()` | Load both train and validation splits at once |

Both functions accept data either as simple file paths or as `FileLoadConfig` objects for per-file perturbation type control.

### Simple Loading (All Types)

```python
from src.llm_finetune.data_loading_utils import load_fact_checking_data

dataset = load_fact_checking_data(
    json_paths=["data/perturbed_data/train_split/numeration_true.json"],
    split="train",
    expand_perturbations_flag=True,
    include_doc=True,
)
```

This loads every perturbation key found in the JSON file (`perturb_type1`, `perturb_type2`, `perturb_type3_*`).

### Per-File Type Filtering (Selective Loading)

Use `FileLoadConfig` to control which perturbation types are extracted from each file:

```python
from src.llm_finetune.data_loading_utils import (
    load_fact_checking_data,
    FileLoadConfig,
    PerturbationType,
)

dataset = load_fact_checking_data(
    file_configs=[
        # Load type1 + type2 from true-labeled claims
        FileLoadConfig(
            path="data/perturbed_data/train_split/numeration_true.json",
            perturb_types=[PerturbationType.PERTURB_TYPE1, PerturbationType.PERTURB_TYPE2],
        ),
        # Load only type1 from false-labeled claims (omit type2 False→True)
        FileLoadConfig(
            path="data/perturbed_data/train_split/numeration_false.json",
            perturb_types=[PerturbationType.PERTURB_TYPE1],
        ),
    ],
    split="train",
)
```

When `perturb_types=None` (the default), all perturbation keys in the file are loaded. The `ORIGINAL_NORMALIZED` entry is **always** included regardless of the filter.

### CLI-Style File Configs

For shell scripts, there is a shorthand syntax parsed by `parse_file_config()`:

```
path/to/file.json                   → Load all types
path/to/file.json:type1,type2       → Load only type1 and type2
path/to/file.json:type3_random      → Load only type3_random
```

Type names accept shorthand — `type1` is automatically expanded to `perturb_type1`. This is used by the training scripts via `--train_json` and `--val_json` arguments.

---

## Perturbation Expansion (`expand_perturbations`)

The `expand_perturbations()` function in `perturbation_logic.py` is the core engine that converts a raw JSON example into multiple rows — one per perturbation variant.

### How It Works

Given a raw JSON entry like:

```json
{
    "claim_id": "abc123",
    "normalized_claim": "The rate increased by 50 percent.",
    "doc": "Evidence text...",
    "label": true,
    "perturb_type1_True": "The rate increased by fifty percent.",
    "perturb_type2_True": "The rate increased by fifty-five percent.",
    "perturb_type3_random_False": "The rate increased by 87 percent."
}
```

`expand_perturbations()` produces:

| # | claim | label | perturbation_type |
|---|---|---|---|
| 1 | The rate increased by 50 percent. | True | `original_normalized` |
| 2 | The rate increased by fifty percent. | True | `perturb_type1` |
| 3 | The rate increased by fifty-five percent. | False | `perturb_type2` |
| 4 | The rate increased by 87 percent. | False | `perturb_type3_random` |

### Key Detection

A regex (`_PERTURB_KEY_RE`) scans each key in the JSON object to find perturbation entries. Recognised key patterns:

- `perturb_type1_True`, `perturb_type1_False`
- `perturb_type2_True`, `perturb_type2_False`
- `perturb_type3_random_True`, `perturb_type3_random_False`
- `perturb_type3_masked_True`, `perturb_type3_masked_False`
- `perturb_type3_removed_True`, `perturb_type3_removed_False`

The `_True`/`_False` suffix on JSON keys indicates the original claim's gold label, **not** the perturbed label. The actual perturbed label is computed by `_determine_perturbed_label()`.

### Label Logic

| Perturbation Type | Label Rule | Rationale |
|---|---|---|
| `perturb_type1` (structure-preserving) | Same as original | Semantically equivalent rewrite |
| `perturb_type2` (value-modifying) | Flipped from original | Numerical value changed → truth value inverts |
| `perturb_type3_*` (obfuscation) | Always `False` | Original value destroyed → unverifiable |

### Filtering

When `allowed_types` is provided, only perturbation keys whose base type matches the filter are included. The original (`ORIGINAL_NORMALIZED`) entry is always emitted:

```python
expanded = expand_perturbations(
    examples,
    allowed_types=[PerturbationType.PERTURB_TYPE1],  # Only structure-preserving
)
```

### Skipping Empty Keys

If a perturbation key is present but its value is empty or `None` (the perturbation module did not produce output for that claim), the entry is silently skipped.

---

## Perturbation Types

Defined in `perturbation_types.py` as the `PerturbationType` enum:

| Enum Member | Value | Category |
|---|---|---|
| `ORIGINAL_NORMALIZED` | `original_normalized` | Baseline (unperturbed) |
| `PERTURB_TYPE1` | `perturb_type1` | Structure-preserving |
| `PERTURB_TYPE2` | `perturb_type2` | Value-modifying |
| `PERTURB_TYPE3_RANDOM` | `perturb_type3_random` | Obfuscation (random digits) |
| `PERTURB_TYPE3_MASKED` | `perturb_type3_masked` | Obfuscation (masked with `#`) |
| `PERTURB_TYPE3_REMOVED` | `perturb_type3_removed` | Obfuscation (numbers removed) |

Convenience sets are provided for grouping:

```python
STRUCTURE_PRESERVING_TYPES = {PerturbationType.PERTURB_TYPE1}
VALUE_MODIFYING_TYPES      = {PerturbationType.PERTURB_TYPE2}
OBFUSCATION_TYPES          = {PERTURB_TYPE3_RANDOM, PERTURB_TYPE3_MASKED, PERTURB_TYPE3_REMOVED}
ALL_PERTURB_TYPES          = STRUCTURE_PRESERVING_TYPES | VALUE_MODIFYING_TYPES | OBFUSCATION_TYPES
```

---

## Conversation Formatting

`convert_to_conversations()` in `conversation.py` transforms expanded examples into the chat format expected by HuggingFace `SFTTrainer`:

```python
{
    "conversations": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "Given the claim and evidence..."},
        {"role": "assistant", "content": '{"label": true}'},
    ],
    "claim_id": "abc123",
    "perturbation_type": "perturb_type1",
    "label": '{"label": true}',
}
```

- The **system prompt** instructs the model to act as a fact-checker.
- The **user prompt** contains the claim and (optionally) the evidence document. Set `include_doc=False` to omit evidence.
- The **assistant response** is the JSON-formatted label (`{"label": true}` or `{"label": false}`).
- The assistant message is always included — SFT needs it for training, and evaluation uses it for scoring.

---

## Dataset Balancing (`DatasetBalancer`)

`class_balancer.py` provides `DatasetBalancer` for controlling label and perturbation distributions in the training set.

### Features

- **Label balancing**: Downsample the majority class (True/False) to match the minority class.
- **Perturbation ratio**: Control the ratio of perturbed to non-perturbed (original) examples. A ratio of `1.0` means equal counts.
- **False perturbation split**: Among False-labeled perturbed examples, control the ratio of True→False (value-modified originals) vs False→False (structure-preserved false claims). Only `perturb_type1` perturbations are used for F→F to ensure they are semantically close to the original.

### Usage

```python
from src.llm_finetune.data_loading_utils.class_balancer import DatasetBalancer

balancer = DatasetBalancer(
    dataset=train_data,
    balance_labels=True,          # Balance True/False distribution
    perturb_ratio=1.0,            # 1:1 perturbed to original
    false_perturb_t_to_f_ratio=0.67,  # 67% T→F, 33% F→F among False perturbations
    seed=42,
)
balanced_dataset = balancer.balance()
```

The balancer categorises every example into one of five bins:

| Category | Description |
|---|---|
| Original True | Unperturbed claims with True label |
| Original False | Unperturbed claims with False label |
| T→T | True-labeled perturbations of True originals (type1) |
| T→F | False-labeled perturbations of True originals (type2, type3) |
| F→F | False-labeled perturbations of False originals (type1 only) |

---

## Batch Sampler (`EnhancedPerturbationBatchSampler`)

`batch_sampler.py` implements `EnhancedPerturbationBatchSampler`, a PyTorch `Sampler` that constructs training batches with awareness of claim structure.

### Motivation

Standard random batching treats each example independently. When fine-tuning on perturbation data, it is beneficial for the model to see an original claim alongside its perturbed variants within the same batch, enabling it to learn the contrast between equivalent, modified, and obfuscated numerical expressions.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `batch_match_claim_id` | `False` | Group examples sharing the same `claim_id` into the same batch |
| `split_large_claims` | `True` | When a claim has more perturbations than `batch_size`, split across multiple batches (original appears only in the first) |
| `ensure_original_in_batch` | `True` | Every batch must contain at least one original (unperturbed) example |
| `ensure_perturbed_in_batch` | `True` | Every batch must contain at least one perturbed example |
| `shuffle` | `True` | Shuffle batch order (and within-claim order) each epoch |
| `drop_last` | `False` | Drop the final batch if it is smaller than `batch_size` |

### Batching Strategies

**Claim-matched batching** (`batch_match_claim_id=True`):
1. Groups all examples by `claim_id`.
2. For each claim, places the original first, followed by perturbations.
3. If the group fits in one batch, it becomes a single batch.
4. If the group is larger than `batch_size` and `split_large_claims=True`, it is split into multiple batches. The original claim only appears in the **first** batch (never repeated).
5. After forming batches, the constraint enforcement pass swaps indices between batches to satisfy the `ensure_original_in_batch` / `ensure_perturbed_in_batch` requirements.

**Mixed batching** (`batch_match_claim_id=False`):
1. All indices are shuffled into a flat list.
2. Batches are formed by slicing consecutive indices.
3. The same constraint enforcement pass is applied if requested.

### Integration with SFTTrainer

The sampler must be created **before** metadata columns (`claim_id`, `perturbation_type`) are removed from the dataset (SFTTrainer requires tensor-only columns). The `PerturbationAwareSFTTrainer` class handles this:

```python
# 1. Build sampler while dataset still has metadata columns
sampler = EnhancedPerturbationBatchSampler(
    dataset=train_data,  # has claim_id, perturbation_type
    batch_size=5,
    batch_match_claim_id=True,
)

# 2. Remove non-tensor columns
train_data = train_data.remove_columns(["claim_id", "perturbation_type", ...])

# 3. Pass pre-built sampler to trainer
trainer = PerturbationAwareSFTTrainer(
    model=model,
    train_dataset=train_data,
    use_custom_sampler=True,
    pre_built_batch_sampler=sampler,
    ...
)
```

---

## Fine-Tuning (`unsloth_finetune.py`)

The training script orchestrates the full pipeline:

1. **Load model** via Unsloth `FastLanguageModel` (4-bit quantised by default).
2. **Apply LoRA** adapters to attention and MLP projection layers.
3. **Set chat template** (Llama-3.1 or ChatML depending on model family).
4. **Load & expand data** via `load_train_and_validation()`.
5. **Balance dataset** via `DatasetBalancer` (optional).
6. **Tokenise** with instruction masking — only the assistant response tokens are trained on (instruction tokens get label `-100`).
7. **Build batch sampler** (if `batch_match_claim_id` is enabled) before removing metadata columns.
8. **Train** with `SFTTrainer` or `PerturbationAwareSFTTrainer`.
9. **Save** LoRA weights and merged 16-bit model.

### CLI Usage

```bash
python -m src.llm_finetune.unsloth_finetune \
    --train_json data/perturbed_data/train_split/numeration_true.json \
                 data/perturbed_data/train_split/numeration_false.json \
    --val_json   data/perturbed_data/val_split/numeration_true.json \
                 data/perturbed_data/val_split/numeration_false.json \
    --model Qwen3-8B \
    --rank 32 \
    --num_train_epochs 3 \
    --batch_size 5 \
    --max_seq_length 8192 \
    --balance_labels \
    --batch_match_claim_id \
    --expand_perturbations
```

### Per-File Type Filtering from CLI

To exclude `type2` perturbations from false-labeled files:

```bash
python -m src.llm_finetune.unsloth_finetune \
    --train_json "data/.../numeration_true.json:type1,type2" \
                 "data/.../numeration_false.json:type1" \
    --val_json   "data/.../numeration_true.json:type1,type2" \
                 "data/.../numeration_false.json:type1" \
    --model Qwen3-8B \
    ...
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_json` | (required) | Training JSON path(s), optionally with `:type` suffix |
| `--val_json` | (required) | Validation JSON path(s) |
| `--model` | — | Model name or HuggingFace path |
| `--rank` | 32 | LoRA rank |
| `--num_train_epochs` | 1 | Number of training epochs |
| `--start_epoch` | 1 | Starting epoch (for resumed training) |
| `--resume_model` | `None` | Path to merged model for resuming (required when `start_epoch > 1`) |
| `--batch_size` | 2 | Per-device batch size |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--max_seq_length` | 16384 | Maximum sequence length in tokens |
| `--expand_perturbations` | `True` | Expand perturbation keys into separate examples |
| `--balance_labels` | `False` | Balance True/False label distribution |
| `--perturb_ratio` | 1.0 | Perturbed-to-original ratio |
| `--false_perturb_t_to_f_ratio` | 0.67 | T→F share among False perturbations |
| `--batch_match_claim_id` | `False` | Enable claim-aware batch grouping |
| `--debug` | `False` | Print conversation format and tokenisation diagnostics |

### Output Structure

```
models/<model_name>_<timestamp>_<perturb_types>/
├── checkpoints_r32/          # Training checkpoints
└── epoch-3/
    ├── lora_weights_r32/     # LoRA adapter weights
    └── merged_model_r32/     # Full merged 16-bit model
```

---

## Inference (`unsloth_inference.py`)

Runs a trained model on test data and computes per-perturbation metrics.

### Model Resolution Order

1. **Direct local path**: If `--model` points to a local directory, use it directly.
2. **Training checkpoint**: If `--start_epoch > 0`, look for a checkpoint at the conventional save path.
3. **HuggingFace Hub**: Fall back to downloading from HuggingFace (with alias resolution for common names).

### Inference Backends

- **Unsloth** (default): Loads with `FastLanguageModel`, generates one example at a time.
- **vLLM** (`--use_vllm`): Batch inference with vLLM for higher throughput.

### Label Extraction

`extract_label_from_response()` uses strict JSON parsing — only a response matching `{"label": true}` or `{"label": false}` is accepted. Qwen3 `<think>...</think>` tokens are stripped before parsing. Responses that cannot be parsed are recorded as parse failures and excluded from metrics.

### CLI Usage

```bash
python -m src.llm_finetune.unsloth_inference \
    --test_json data/perturbed_data/test_split/numeration_true.json \
                data/perturbed_data/test_split/numeration_false.json \
    --model Qwen3-8B \
    --rank 32 \
    --start_epoch 3 \
    --max_seq_length 8192 \
    --expand_perturbations \
    --output_dir predictions/numpert
```

### Output Files

```
predictions/<timestamp>/<model>-epoch-<N>/
├── inference_details.json   # Full prompts + raw model outputs
├── predictions.csv          # Structured predictions table
├── metrics.json             # Accuracy, F1, per-perturbation breakdown
└── parse_failures.json      # Unparseable responses (for debugging)
```

---

## Sequence Length Analysis

`sequence_len_utility.py` provides `analyze_sequence_lengths()` to profile the token distribution of your dataset before training. It reports min/max/mean/median/p95/p99 sequence lengths and saves a histogram plot. Use it to choose an appropriate `--max_seq_length`:

```python
from src.llm_finetune.data_loading_utils.sequence_len_utility import analyze_sequence_lengths

stats = analyze_sequence_lengths(
    json_paths=["data/perturbed_data/train_split/numeration_true.json"],
    tokenizer=tokenizer,
    split="train",
)
# Prints recommended max_seq_length based on 99th percentile
```

---

## Shell Script Configs

Pre-configured training scripts are in `scripts/train/model_configs/`. Each script sets all training hyperparameters and data paths, then invokes `unsloth_finetune.py`. Two batching modes are provided:

| Script | Batching | Notes |
|---|---|---|
| `qwen3_8b_enhanced_b.sh` | Claim-matched (`batch_match_claim_id=true`) | Groups original + perturbations per batch |
| `qwen3_8b_normal_b.sh` | Random (standard) | No claim grouping |
| `qwen3_8b_normal_b_no_type2f.sh` | Random + per-file filtering | Excludes `type2` from false-labeled files |

Submit via SLURM:

```bash
sbatch scripts/train/start_8b_enchanced_batch.sh
```
