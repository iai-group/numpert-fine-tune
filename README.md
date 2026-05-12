![coverage](https://img.shields.io/badge/dynamic/json?url=https://gist.githubusercontent.com/vinaysetty/4dc000699f87ec295368ae52250acc7a/raw/coverage.numpert-fine-tune.main.json&label=coverage&query=$.message&color=$.color)



# LLM Fine-Tuning Module

This module implements the full fine-tuning and inference pipeline for numerical perturbation-aware fact-checking models. It uses [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning and supports custom batch sampling strategies that group original and perturbed claims together.

---

## Directory Structure

```
src/llm_finetune/
├── batch_sampler.py               # Perturbation-aware batch sampler (legacy, see note below)
├── config.py                      # Path prefix configuration
├── unsloth_finetune.py            # Training entry point
├── unsloth_inference.py           # Inference entry point
├── upload_model_hf.py             # Upload trained model to HuggingFace Hub
├── utils/
│   ├── data_loading/
│   │   ├── pipeline.py                # Unified entry point: load_split()
│   │   ├── dataset_balancer.py        # Core engine: load, flatten, deduplicate, balance, couple, shuffle
│   │   ├── perturbation_types.py      # PerturbationType enum, FileLoadConfig, parse_file_config()
│   │   ├── conversation.py            # Convert examples to SFTTrainer chat format
│   │   └── sequence_len_utility.py    # Sequence length analysis & plotting
│   ├── training/
│   │   ├── convert_to_gguf.py         # GGUF quantization (in-pipeline & standalone)
│   │   └── training_config_export.py  # Save training params & data provenance to JSON
│   └── inference/
│       ├── inference_config_export.py # Save inference run parameters to JSON
│       ├── metrics.py                 # Structured metrics computation & reporting
│       └── unperturbed_claims.py      # Deduplicate unperturbed claims across files
```

---

## Data Loading Pipeline

The data loading pipeline converts raw perturbed JSON files into a HuggingFace `Dataset` ready for `SFTTrainer`. All data loading goes through the unified `load_split()` function in `pipeline.py`, which always routes through `dataset_balancer` as the single core engine.

### Pipeline Flow

```
Raw JSON files
  → build_balanced_dataset()            # Load, flatten, deduplicate, balance, couple, shuffle
  → _enrich_samples()                   # Attach evidence docs & perturbation type labels
  → convert_to_conversations()          # Format as system/user/assistant chat messages
  → Dataset.from_list()                 # HuggingFace Dataset
```

When no balancing flags are set, the balancer still performs load → flatten/expand → deduplicate → couple → shuffle, just without any under-/over-sampling. This means the same code path is used regardless of whether data balancing is requested.

### Entry Point

The primary entry point is `load_split()` in `pipeline.py`:

```python
from src.llm_finetune.utils.data_loading.pipeline import load_split

dataset = load_split(
    file_configs=configs,      # FileLoadConfig list (path + optional type filter)
    json_paths=paths,          # or simple path list (mutually exclusive with file_configs)
    split="train",
    use_perturbations=True,
    include_doc=True,
    prompt_type=PromptType.ZERO_SHOT,  # ZERO_SHOT | FEW_SHOT | NEGATIVE_SYNTHETIC_ALL_WITH_REASONING
    balance_labels=False,
    perturb_ratio=None,        # None = keep all perturbations; float = target ratio
    false_perturb_t_to_f_ratio=None,
    method_balance_mode=None,     # None/"none" = no method balancing; "undersample" or "oversample"
    seed=3407,
)
```

Per-file perturbation type filtering (via `FileLoadConfig.perturb_types`) is respected. `load_split()` builds a `{filename: {allowed_types}}` mapping from the configs and passes it to the balancer's `flatten_samples()`, which filters perturbation keys during record expansion.


### Simple Loading (All Types)

```python
from src.llm_finetune.utils.data_loading.pipeline import load_split

dataset = load_split(
    json_paths=["data/perturbed_data/train_split/numeration_true.json"],
    split="train",
    use_perturbations=True,
    include_doc=True,
)
```

This loads every perturbation key found in the JSON file (`perturb_type1`, `perturb_type2`, `perturb_type3_*`).

### Per-File Type Filtering (Selective Loading)

Use `FileLoadConfig` to control which perturbation types are extracted from each file:

```python
from src.llm_finetune.utils.data_loading.pipeline import load_split
from src.llm_finetune.utils.data_loading.perturbation_types import (
    FileLoadConfig,
    PerturbationType,
)

dataset = load_split(
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

## Perturbation Types

Defined in `src.llm_finetune.utils.data_loading.perturbation_types` as the `PerturbationType` enum:

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

`convert_to_conversations()` in `src.llm_finetune.utils.data_loading.conversation` transforms expanded examples into the chat format expected by HuggingFace `SFTTrainer`:

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
- The **`prompt_type`** parameter controls which example block is injected before the claim (see [Prompt Types](#prompt-types) below).
- The **assistant response** is the JSON-formatted label (`{"label": true}` or `{"label": false}`).
- The assistant message is always included — SFT needs it for training, and evaluation uses it for scoring.

---

## Prompt Types

The user prompt injected into each conversation is controlled by the `prompt_type` parameter, which accepts a `PromptType` enum value (defined in `src.numpert_prompt.prompt`). All three types use the same base template (`FACT_CHECK_USER_PROMPT`) — they differ only in the example block prepended before the claim.

| `PromptType` | CLI string | Injected content |
|---|---|---|
| `ZERO_SHOT` | `ZERO_SHOT` | No examples (default) |
| `FEW_SHOT` | `FEW_SHOT` | Two full-length true/false examples with evidence |
| `NEGATIVE_SYNTHETIC_ALL_WITH_REASONING` | `NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING` | Six synthetic numerical perturbation examples, each with a `Reasoning:` chain explaining why the claim is false |

The `prompt_type` flows through the entire data loading pipeline:

```
load_split(prompt_type=...) → convert_to_conversations(prompt_type=...) → get_fact_check_user_prompt(prompt_type=...)
```

`prompt_type` only affects the **user turn** — the system prompt and assistant response are unchanged.

---

## Dataset Balancing (`build_balanced_dataset`)

`src.llm_finetune.utils.data_loading.dataset_balancer` provides `build_balanced_dataset()` — the core engine that all data loading flows through. It handles loading, expansion, deduplication, coupling, and (optionally) balancing in one pass.

### Features

- **Label balancing**: Downsample the majority class (True/False) to match the minority class.
- **Perturbation ratio**: Control the ratio of perturbed to non-perturbed (original) examples. `None` (the default in `load_split`) keeps all perturbations; `1.0` means equal counts.
- **True perturbation fraction**: Control the share of perturbed examples that remain True (T→T) vs produce False (T→F, F→F).
- **False perturbation split**: Among False-labeled perturbed examples, control the share that are T→F versus F→F.
- **Method balancing**: Undersample or oversample so each perturbation method contributes equally within each perturbation type.
- **Cross-file fairness**: Undersampling and oversampling use round-robin across source files to avoid exhausting any single file.
- **Coupling**: Ensures that for every selected perturbation, the corresponding original claim is also included.
- **Per-file type filtering**: When `FileLoadConfig.perturb_types` is set, only the specified perturbation types are extracted from that file.

### Usage

The balancer is invoked automatically by `load_split()`. Direct usage:

```python
from src.llm_finetune.utils.data_loading.dataset_balancer import (
    build_balanced_dataset,
)

result = build_balanced_dataset(
    file_paths=["data/.../numeration_True.json", "data/.../numeration_False.json"],
    balance_labels=True,
    perturb_ratio=1.0,  # explicitly set; default is None (keep all)
    false_perturb_t_to_f_ratio=0.5,
  method_balance_mode="undersample",
    seed=42,
    allowed_types_by_file={"numeration_False.json": {"type1"}},  # optional per-file filtering
)
```

### Pipeline

The balancer operates in a fixed priority pipeline:

1. **Load** — Read JSON files into (filename, records) pairs.
2. **Flatten** — Expand each record into original + perturbation `Sample` objects (applying per-file type filtering here).
3. **Deduplicate** — Remove samples with identical text.
4. **Balance** — Apply constraints in order (method balance → ratio → label balance), with higher-priority constraints getting the final say.
5. **Couple** — Ensure originals are present for every selected perturbation.
6. **Shuffle** — Deterministic shuffle based on seed.

### Sample Categories

The balancer categorises every example into one of four types:

| Category | Description |
|---|---|
| Original | Unperturbed claims (always included for coupling) |
| T→T | True-labeled perturbations of True originals (type1) |
| T→F | False-labeled perturbations of True originals (type2, type3) |
| F→F | False-labeled perturbations of False originals (type1 only) |

---

## Batch Sampler (`EnhancedPerturbationBatchSampler`)

> **Note:** The batch sampler is legacy code that is not currently in active use. The claim-grouping logic does not work as intended with the current training pipeline and may produce unexpected batch compositions. It is retained for reference but should not be relied upon for production training runs.

`src.llm_finetune.batch_sampler` implements `EnhancedPerturbationBatchSampler`, a PyTorch `Sampler` that constructs training batches with awareness of claim structure.

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
4. **Load & expand data** via `load_split()` (calls `pipeline.load_split()` for both train and validation splits).
5. **Tokenise** with instruction masking — only the assistant response tokens are trained on (instruction tokens get label `-100`).
6. **Build batch sampler** (if `batch_match_claim_id` is enabled) before removing metadata columns.
7. **Train** with `SFTTrainer` or `PerturbationAwareSFTTrainer`.
8. **Save** LoRA weights and merged 16-bit model.

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
    --use_perturbations
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
| `--use_perturbations` | `True` | Loads perturbation into separate examples |
| `--balance_labels` | `False` | Balance True/False label distribution |
| `--perturb_ratio` | `None` | Ratio of perturbed to original examples; `None` disables the constraint (keep all) |
| `--false_perturb_t_to_f_ratio` | `None` | Among False perturbed examples, fraction that are T→F flips (e.g. `0.67`); `None` disables the constraint |
| `--method_balance_mode` | `None` | Balance perturbation methods: `undersample`, `oversample`, or `none` (omit to disable) |
| `--batch_match_claim_id` | `False` | Enable claim-aware batch grouping |
| `--split_large_claims` | `True` | Split claims with more examples than the batch size across multiple batches (requires `--batch_match_claim_id`) |
| `--ensure_original_in_batch` | `True` | Guarantee each batch contains at least one original example (requires `--batch_match_claim_id`) |
| `--ensure_perturbed_in_batch` | `True` | Guarantee each batch contains at least one perturbed example (requires `--batch_match_claim_id`) |
| `--eval_steps` | 500 | Evaluation frequency; float in `(0, 1]` is treated as a fraction of total training steps |
| `--save_steps` | 500 | Checkpoint save frequency (steps) |
| `--save_gguf` | `False` | Also save a GGUF-quantized version of the final model |
| `--gguf_quantization_method` | `q8_0` | GGUF quantization method (e.g. `q4_k_m`) |
| `--debug` | `False` | Print conversation format and tokenisation diagnostics |

### Output Structure

```
models/<model_name>_<timestamp>_<perturb_types>/
├── training_config.json      # Full training provenance (see below)
├── checkpoints_r32/          # Training checkpoints
└── epoch-3/
    ├── lora_weights_r32/     # LoRA adapter weights
    ├── merged_model_r32/     # Full merged 16-bit model
    └── gguf_q4_k_m_r32/      # (optional) GGUF quantized model
```

#### `training_config.json`

A machine-readable JSON file written to the model output directory at the start of every training run. It records the exact parameters, data files, perturbation types, and dataset statistics used, so every trained model carries full provenance. The file is produced by `src.llm_finetune.utils.training.training_config_export` and has the following structure:

```json
{
  "timestamp": "2025-06-15T12:00:00",
  "model": {
    "name": "unsloth/Qwen3-8B",
    "lora_rank": 32
  },
  "training": {
    "start_epoch": 1,
    "num_train_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "max_seq_length": 8192,
    "learning_rate": 0.0001
  },
  "data": {
    "use_perturbations": true,
    "balance_labels": true,
    "perturb_ratio": null,
    "false_perturb_t_to_f_ratio": 0.67,
    "train_files": [
      {
        "path": "data/perturbed_data/train_split/numeration_True.json",
        "perturb_types": ["perturb_type1", "perturb_type2"]
      },
      {
        "path": "data/perturbed_data/train_split/numeration_False.json",
        "perturb_types": null
      }
    ],
    "val_files": [
      {
        "path": "data/perturbed_data/val_split/numeration_True.json",
        "perturb_types": null
      }
    ],
    "train_stats": {
      "size": 1200
    },

    "val_stats": {
      "size": 200,
    }
  },
  "batch_sampler": {
    "batch_match_claim_id": true,
    "split_large_claims": true,
    "ensure_original_in_batch": true,
    "ensure_perturbed_in_batch": true
  },
  "gguf": {
    "save_gguf": false,
    "quantization_method": "q4_k_m"
  }
}
```

Key details:
- **`data.train_files` / `data.val_files`**: Each entry records the file path and the perturbation types loaded from it. A value of `null` for `perturb_types` means all types present in the file were loaded.
- **`data.train_stats` / `data.val_stats`**: Computed after any balancing or filtering, so they reflect the actual data the model trained on.
- The file is generated by `build_training_config()` and written by `save_training_config()` from `src.llm_finetune.utils.training.training_config_export`.

---

## GGUF Quantization (`convert_to_gguf`)

GGUF is the format used by [llama.cpp](https://github.com/ggerganov/llama.cpp) and tools like [Ollama](https://ollama.com/) for efficient CPU/GPU inference. The module (`src.llm_finetune.utils.training.convert_to_gguf`) supports saving GGUF models in two ways.

### During Training

Pass `--save_gguf` to the training script to automatically export a GGUF model after training completes. You can choose the quantization method with `--gguf_quantization_method`:

```bash
python -m src.llm_finetune.unsloth_finetune \
    --train_json data/perturbed_data/train_split/numeration_True.json \
    --val_json   data/perturbed_data/val_split/numeration_True.json \
    --model Qwen3-8B \
    --save_gguf \
    --gguf_quantization_method q8_0
```

The GGUF output is saved alongside the merged model:

```
models/<model_name>_<timestamp>_<perturb_types>/epoch-3/
├── merged_model_r32/
└── gguf_q8_0_r32/      # ← GGUF output
```

### After Training (Standalone)

Convert any existing merged model to GGUF using the standalone CLI:

```bash
python -m src.llm_finetune.utils.training.convert_to_gguf \
    --model_path models/.../epoch-3/merged_model_r32 \
    --quantization_method q4_k_m
```

If `--output_dir` is not specified, the GGUF model is saved to `<model_path>_gguf_<quant_method>/`.

You can also call `convert_existing_model()` from Python:

```python
from src.llm_finetune.utils.training.convert_to_gguf import convert_existing_model

convert_existing_model(
    model_path="models/.../epoch-3/merged_model_r32",
    quantization_method="q5_k_m",
)
```

### Supported Quantization Methods

| Method | Bits | Size vs f16 | Quality | Use Case |
|---|---|---|---|---|
| `f16` | 16 | 1× | Lossless | Full-precision inference |
| `q8_0` | 8 | ~0.5× | Near-lossless | High-quality, moderate memory |
| `q6_k` | 6 | ~0.38× | Very good | Good balance of quality and size |
| `q5_k_m` | 5 | ~0.31× | Good | Recommended for most uses |
| `q5_0` | 5 | ~0.31× | Good | Simpler 5-bit scheme |
| `q4_k_m` | 4 | ~0.25× | Good (default) | Best size/quality tradeoff |
| `q4_0` | 4 | ~0.25× | Acceptable | Fastest 4-bit |
| `q3_k_m` | 3 | ~0.19× | Acceptable | Memory constrained |
| `q2_k` | 2 | ~0.13× | Lower | Extreme compression |

---

## Inference (`unsloth_inference.py`)

Runs a trained model on test data and computes per-perturbation metrics.

### Model Resolution Order

1. **Direct local path**: If `--model` points to a local directory, use it directly.
2. **Training checkpoint**: If `--start_epoch > 0`, look for a checkpoint at the conventional save path.
3. **HuggingFace Hub**: Fall back to downloading from HuggingFace (with alias resolution for common names).

### Inference Backend

- **Unsloth**: Loads with `FastLanguageModel`, generates one example at a time.

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
    --use_perturbations \
    --prompt_type ZERO_SHOT \
    --output_dir predictions/numpert
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--test_json` | (required) | Test JSON path(s), optionally with `:type` suffix |
| `--model` | — | Model name, HuggingFace path, local directory, or `.gguf` file |
| `--tokenizer` | `None` | HuggingFace tokenizer path. Required when `--model` is a GGUF file |
| `--rank` | `32` | LoRA rank (used to locate training checkpoints) |
| `--start_epoch` | `0` | Epoch of the checkpoint to load; `0` loads from HuggingFace |
| `--max_seq_length` | `16384` | Maximum sequence length in tokens |
| `--use_perturbations` / `--no-use_perturbations` | `True` | Include or exclude perturbed examples |
| `--inference_unperturbed_claims` | `False` | Run inference only on original (normalized) claims; outputs prefixed with `unperturbed_` |
| `--prompt_type` | `ZERO_SHOT` | User prompt variant: `ZERO_SHOT`, `FEW_SHOT`, or `NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING` |
| `--output_dir` | (auto) | Output directory; auto-generates a timestamped run name if omitted |
| `--resume` | `False` | Resume a previous run from an existing `inference_details.jsonl` |
| `--verbose` | `False` | Enable verbose llama.cpp output |
| `--require-gpu-offload` | `False` | Fail hard if llama-cpp-python was built without GPU support |

### Output Files

```
predictions/<timestamp>/<model>-epoch-<N>/
├── inference_config.json    # Model, data files, and run parameters
├── inference_details.json   # Full prompts + raw model outputs
├── predictions.csv          # Structured predictions table
├── metrics.json             # Accuracy, F1, per-perturbation breakdown
└── parse_failures.json      # Unparseable responses (for debugging)
```

---

## Metrics (`utils/inference/metrics.py`)

The metrics module computes and reports classification scores with breakdowns at multiple levels of granularity. It is separated from the inference script so that metrics can be recomputed, extended, or consumed independently.

### Report Structure

`compute_metrics()` produces a `MetricsReport` containing four sections:

| Section | Grouping | Purpose |
|---|---|---|
| **Overall** | All examples combined | Single aggregate accuracy / F1 |
| **Per perturbation type** | One group per `perturbation_type` value | See how the model handles each perturbation independently |
| **Per source file** | One group per input JSON file | Compare performance across different data files (e.g. `approximation_True.json` vs `numeration_False.json`) |
| **Per file + type** | Cross-product of source file × perturbation type | Most granular view — e.g. accuracy on `perturb_type2` examples that came from `approximation_True.json` |

Each section is optional and can be toggled via boolean flags:

```python
report = compute_metrics(
    results,
    include_per_type=True,           # default: True
    include_per_file=True,           # default: True
    include_per_file_and_type=True,  # default: True
)
```

Per-file breakdowns require a `source_file` field to be present in the result records (automatically set by the data loading pipeline).

### Scores per Group

Every group (overall and each sub-group) is represented by a `ClassificationScores` dataclass containing:

| Field | Description |
|---|---|
| `accuracy` | Standard accuracy (correct / valid) |
| `f1_macro` | Macro-averaged F1 score |
| `f1_weighted` | Weighted F1 score |
| `precision_macro` | Macro-averaged precision |
| `recall_macro` | Macro-averaged recall |
| `total` | Total number of examples in the group |
| `valid` | Examples successfully parsed (`total - parse_failures`) |
| `correct` | Number of correct predictions |
| `parse_failures` | Responses that could not be parsed as valid JSON |
| `classification_report_str` | Full sklearn `classification_report` text |

Parse failures (responses that didn't produce valid `{"label": true/false}` JSON) are **excluded** from metric computation but tracked separately per group.

### Console Output

`print_metrics_report()` prints a structured summary to stdout:

```
================================================================================
  OVERALL RESULTS  (150/152 parseable responses)
================================================================================
  Accuracy:       0.8533
  F1 Macro:       0.8521
  F1 Weighted:    0.8537
  Precision (M):  0.8600
  Recall (M):     0.8467
  Parse failures: 2/152

================================================================================
  METRICS BY PERTURBATION TYPE
================================================================================
  Group                                      Acc    F1-M    F1-W    Prec     Rec     N  Fail
  -----------------------------------------------------------------------------------------
  original_normalized                      0.9000  0.8980  0.9010  0.9100  0.8900    50     0
  perturb_type1                            0.8800  0.8756  0.8812  0.8900  0.8600    50     1
  perturb_type2                            0.7800  0.7789  0.7802  0.7800  0.7900    50     1

================================================================================
  METRICS BY SOURCE FILE
================================================================================
  Group                                      Acc    F1-M    F1-W    Prec     Rec     N  Fail
  -----------------------------------------------------------------------------------------
  approximation_True.json                  0.8667  0.8650  0.8670  0.8700  0.8600    76     1
  numeration_False.json                    0.8378  0.8350  0.8390  0.8500  0.8300    74     1

================================================================================
  METRICS BY SOURCE FILE + PERTURBATION TYPE
================================================================================
  Group                                      Acc    F1-M    F1-W    Prec     Rec     N  Fail
  -----------------------------------------------------------------------------------------
  approximation_True.json | original_norm  0.9200  ...
  approximation_True.json | perturb_type1  0.8800  ...
  numeration_False.json | perturb_type2    0.7600  ...
  ...
```

Pass `show_classification_reports=True` to also print the full sklearn classification report for each group.

### JSON Output

`save_metrics_report()` writes `metrics.json` with the same structure:

```json
{
  "overall": {
    "group_name": "overall",
    "accuracy": 0.8533,
    "f1_macro": 0.8521,
    "f1_weighted": 0.8537,
    "precision_macro": 0.86,
    "recall_macro": 0.8467,
    "total": 152,
    "valid": 150,
    "correct": 128,
    "parse_failures": 2
  },
  "per_perturbation_type": {
    "original_normalized": { "accuracy": 0.9, "..." : "..." },
    "perturb_type1": { "accuracy": 0.88, "..." : "..." },
    "perturb_type2": { "accuracy": 0.78, "..." : "..." }
  },
  "per_source_file": {
    "approximation_True.json": { "accuracy": 0.8667, "..." : "..." },
    "numeration_False.json": { "accuracy": 0.8378, "..." : "..." }
  },
  "per_file_and_type": {
    "approximation_True.json | original_normalized": { "..." : "..." },
    "approximation_True.json | perturb_type1": { "..." : "..." },
    "numeration_False.json | perturb_type2": { "..." : "..." }
  }
}
```

### Programmatic Usage

The metrics module can be used independently of the inference script:

```python
from src.llm_finetune.utils.inference.metrics import (
    compute_metrics,
    print_metrics_report,
    save_metrics_report,
)

# Each result dict needs at minimum: label, prediction, perturbation_type
results = [
    {
        "claim_id": "c1",
        "label": '{"label": true}',
        "prediction": '{"label": true}',
        "perturbation_type": "original_normalized",
        "source_file": "approximation_True.json",  # optional, enables per-file breakdown
    },
    # ...
]

report = compute_metrics(results)
print_metrics_report(report, show_classification_reports=True)
save_metrics_report(report, output_dir="results/my_experiment")
```

### Integration with Inference

During inference, the pipeline automatically:

1. Calls `build_result_records()` to merge dataset metadata with parsed predictions.
2. Calls `compute_metrics()` to produce the full report.
3. Calls `print_metrics_report()` to display results.
4. Calls `save_metrics_report()`, `save_parse_failures()`, and `save_predictions_csv()` to persist outputs.

### Source File Tracking

The data loading pipeline (`load_split()`) annotates every expanded example with a `source_file` field containing the JSON filename (e.g. `approximation_True.json`). This field flows through the conversation formatter and into the HuggingFace dataset, enabling per-file metric breakdowns without any extra configuration.

---

## Sequence Length Analysis

`src.llm_finetune.utils.data_loading.sequence_len_utility` provides `analyze_sequence_lengths()` to profile the token distribution of your dataset before training. It reports min/max/mean/median/p95/p99 sequence lengths and saves a histogram plot. Use it to choose an appropriate `--max_seq_length`:

```python
from src.llm_finetune.utils.data_loading.sequence_len_utility import analyze_sequence_lengths

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
