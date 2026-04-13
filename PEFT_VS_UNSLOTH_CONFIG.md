# PEFT vs Unsloth Configuration

This document explains the differences between PEFT and Unsloth experiment configurations.

## Model Identifier Differences

### Unsloth (`_set_defaults`)
- Uses short model names: `Qwen3-8B`, `Qwen3-32B`, `llama-3.2-3b`, etc.
- Has `get_model_name_arg()` function that converts short names to full HuggingFace paths
- Examples:
  - `Qwen3-8B` → `unsloth/Qwen3-8B`
  - `llama-3.2-3b` → `unsloth/Llama-3.2-3B-Instruct`

### PEFT (`_set_peft_defaults`)
- Uses full HuggingFace model identifiers: `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, etc.
- No model name resolution - passes identifier directly to HuggingFace Hub
- Must use complete paths to work correctly

## Configuration Defaults

### Unsloth (`_set_defaults`)
```bash
MODEL="Qwen3-8B"              # Short name resolved via get_model_name_arg()
RANK=32
NUM_TRAIN_EPOCHS=2
BATCH_SIZE=5
GRAD_ACCUM=1
```

### PEFT (`_set_peft_defaults`)
```bash
MODEL="Qwen/Qwen3-8B"         # Full HuggingFace identifier
RANK=32                        # For compatibility, though not used
PEFT_METHOD="lora"            # PEFT-specific method
PEFT_RANK=32                  # LoRA rank for PEFT
PEFT_VIRTUAL_TOKENS=20        # For prompt/prefix tuning
NUM_TRAIN_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=1
```

## Available Model Identifiers

### For PEFT Experiments
Use full HuggingFace paths:
- `Qwen/Qwen3-8B` (8B parameters)
- `Qwen/Qwen3-32B` (32B parameters)
- `meta-llama/Llama-3.2-3B-Instruct` (3B parameters)
- `meta-llama/Llama-3.2-1B-Instruct` (1B parameters)

Or any other model available on HuggingFace Hub.

### For Unsloth Experiments
Use short names (automatically resolved):
- `Qwen3-8B` (resolves to `unsloth/Qwen3-8B`)
- `Qwen3-32B` (resolves to `unsloth/Qwen3-32B`)
- `llama-3.2-3b` (resolves to `unsloth/Llama-3.2-3B-Instruct`)
- `llama-3.2-1b` (resolves to `unsloth/Llama-3.2-1B-Instruct`)

Or use full paths directly.

## Customizing Model for PEFT Experiments

To use a different model in PEFT experiments, create a new experiment function:

```bash
exp_peft_qwen3_32b_lora() {
    _set_peft_defaults
    MODEL="Qwen/Qwen3-32B"        # Use full HuggingFace identifier
    PEFT_METHOD="lora"
    PEFT_RANK=64                  # Larger rank for larger model
    NUM_TRAIN_EPOCHS=3
}
```

Then run:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_lora
```

## Why the Difference?

- **Unsloth** has custom model resolution because it maintains optimized versions of models specifically for fast fine-tuning
- **PEFT** is a generic library that works with any HuggingFace model, so it requires the official model identifiers

## Summary

| Aspect | Unsloth | PEFT |
|--------|---------|------|
| **Model Identifier** | Short name or full path | Full HuggingFace path only |
| **Default Model** | `Qwen3-8B` | `Qwen/Qwen3-8B` |
| **Resolution** | Via `get_model_name_arg()` | Direct HuggingFace Hub lookup |
| **Config Function** | `_set_defaults()` | `_set_peft_defaults()` |
| **PEFT Method** | Not supported | `lora`, `prompt_tuning`, `prefix_tuning`, combinations |
| **Training Framework** | Unsloth optimized | Hugging Face Transformers + PEFT |
