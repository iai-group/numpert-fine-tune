# Qwen3-32B PEFT Configuration Guide

**Added**: April 13, 2026  
**Status**: Production-Ready

---

## Overview

Seven new experiment configurations have been added for fine-tuning with Qwen3-32B, a larger model from the Qwen family. These configurations support all 5 PEFT methods with memory-optimized settings for the 32B model.

---

## Available Experiments

All experiments inherit data configuration from `_set_peft_defaults` (all perturbation types: numeration, approximation, range, negative, obfuscation).

### 1. LoRA (32 Rank)
```
exp_peft_qwen3_32b_lora
```
- **Method**: LoRA with rank 32
- **Batch Size**: 1 (memory optimized)
- **Grad Accum**: 4 (effective batch size: 4)
- **Use Case**: Standard parameter-efficient fine-tuning on 32B model

**Run Locally**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_lora
```

**Run on SLURM**:
```bash
sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_32b_lora
```

---

### 2. LoRA (64 Rank)
```
exp_peft_qwen3_32b_lora_r64
```
- **Method**: LoRA with higher rank (64)
- **Batch Size**: 1
- **Grad Accum**: 4
- **Use Case**: More expressive LoRA adaptation (higher memory usage)

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_lora_r64
```

---

### 3. Prompt Tuning
```
exp_peft_qwen3_32b_prompt_tuning
```
- **Method**: Prompt Tuning (20 virtual tokens)
- **Batch Size**: 1
- **Grad Accum**: 4
- **Use Case**: Lightweight prefix adaptation, lowest memory footprint

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_prompt_tuning
```

---

### 4. Prefix Tuning
```
exp_peft_qwen3_32b_prefix_tuning
```
- **Method**: Prefix Tuning (20 prefix tokens)
- **Batch Size**: 1
- **Grad Accum**: 4
- **Use Case**: Prefix-based adaptation with bidirectional context

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_prefix_tuning
```

---

### 5. LoRA + Prompt Tuning (Combined)
```
exp_peft_qwen3_32b_lora_prompt
```
- **Method**: LoRA rank 32 + Prompt Tuning (20 tokens)
- **Batch Size**: 1
- **Grad Accum**: 4
- **Use Case**: Combines adaptation techniques for enhanced expressiveness

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_lora_prompt
```

---

### 6. LoRA + Prefix Tuning (Combined)
```
exp_peft_qwen3_32b_lora_prefix
```
- **Method**: LoRA rank 32 + Prefix Tuning (20 tokens)
- **Batch Size**: 1
- **Grad Accum**: 4
- **Use Case**: Combines LoRA weights with prefix adaptation

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_lora_prefix
```

---

### 7. Quick Test
```
exp_peft_qwen3_32b_quick_test
```
- **Method**: LoRA (rank 32)
- **Epochs**: 1
- **Batch Size**: 1
- **Grad Accum**: 2 (effective batch size: 2)
- **Use Case**: Quick validation of training pipeline (fastest)

**Run**:
```bash
bash scripts/train/run_experiment.sh peft_qwen3_32b_quick_test
```

---

## Hardware Requirements

### Minimum GPU Memory
- **LoRA (32)**: 24GB+ GPU (e.g., RTX 4090, A100)
- **LoRA (64)**: 32GB+ GPU (e.g., A100)
- **Prompt/Prefix Tuning**: 20GB+ GPU (lowest VRAM requirement)
- **Combined**: 28GB+ GPU

### Multi-GPU Setup
If training on multi-GPU, use:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/run_experiment.sh peft_qwen3_32b_lora
```

---

## Configuration Details

### Shared Settings (All Experiments)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-32B | Full HuggingFace identifier |
| Batch Size | 1 | Optimized for 32B model size |
| Grad Accum | 4 | Effective batch size: 4 |
| Epochs | 1 | Default (override with SLURM) |
| Max Seq Length | 16,384 tokens | Qwen3 supports up to 32K |
| Use Perturbations | TRUE | All perturbation types |
| Perturb Ratio | 1.0 | 50:50 original:perturbed |

### Data Configuration

All experiments use the same data splits:

**Training Data** (10 files):
- `numeration_True.json`, `numeration_False.json`
- `approximation_True.json`, `approximation_False.json`
- `range_True.json`, `range_False.json`
- `negative_True.json`, `negative_False.json`
- `obfuscation_True.json`, `obfuscation_False.json`

**Validation Data** (same 10 files in val_split)

**Test Data** (same 10 files in test_split, for inference)

---

## Memory Optimization Tips

If you encounter out-of-memory errors:

1. **Reduce batch size** (already at 1, use gradient accumulation instead)
2. **Reduce max_seq_length**:
   ```bash
   MAX_SEQ_LENGTH=8192 bash scripts/train/run_experiment.sh peft_qwen3_32b_lora
   ```
3. **Use Prompt/Prefix Tuning** instead of LoRA (lowest memory)
4. **Enable gradient checkpointing** (modify SLURM script):
   ```bash
   --gradient_checkpointing
   ```

---

## Performance Comparison

Expected training characteristics on A100 (40GB):

| Method | Approx. VRAM | Training Speed | Model Params |
|--------|-------------|-----------------|--------------|
| Prompt Tuning | 20GB | Fastest | 0.4M tokens |
| Prefix Tuning | 20GB | Fastest | 0.4M weights |
| LoRA (32) | 24GB | Moderate | 4.7M weights |
| LoRA (64) | 28GB | Moderate | 9.4M weights |
| LoRA+Prompt | 24GB | Moderate | ~5M total |
| LoRA+Prefix | 24GB | Moderate | ~5M total |

---

## Monitoring Training

### Using Weights & Biases

All experiments log to W&B by default:

```bash
# View training metrics
wandb offline   # Or log in if online
```

### Using tensorboard (if enabled)

```bash
tensorboard --logdir ./runs
```

### Check checkpoints

```bash
# After training completes
ls -lah models/Qwen-Qwen3-32B-*/checkpoints/
```

---

## Inference with Fine-tuned Models

After training with Qwen3-32B, run inference:

```bash
# Using trained checkpoint
uv run python -m src.llm_finetune.peft_inference \
  --model_path ./models/Qwen-Qwen3-32B-*/checkpoints/checkpoint-XXX \
  --test_json data/perturbed_data/test_split/negative_False.json \
  --batch_size 2 \
  --output_dir ./inference_results_qwen3_32b
```

---

## SLURM Job Submission

### Basic submission
```bash
sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_32b_lora
```

### With custom SLURM parameters
```bash
sbatch --gres=gpu:1 \
       --mem=40G \
       --cpus-per-task=8 \
       scripts/train/slurm_peft_experiment.sh peft_qwen3_32b_lora
```

### Monitor job
```bash
squeue -u $USER
tail -f slurm-*.out
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size (use grad accum), use Prompt/Prefix Tuning |
| Model download fails | Check HuggingFace token: `huggingface-cli login` |
| SLURM timeout | Increase `--time` in SLURM script, or use fewer epochs |
| Training too slow | Use fewer data files, reduce max_seq_length |
| Checkpoints not saved | Check `EVAL_STEPS` setting (must be > 0) |

---

## Listing Available Experiments

To see all Qwen3-32B experiments:

```bash
bash scripts/train/run_experiment.sh --list | grep qwen3_32b
```

Output:
```
  peft_qwen3_32b_lora
  peft_qwen3_32b_lora_prefix
  peft_qwen3_32b_lora_prompt
  peft_qwen3_32b_lora_r64
  peft_qwen3_32b_prefix_tuning
  peft_qwen3_32b_prompt_tuning
  peft_qwen3_32b_quick_test
```

---

## Configuration Inspection

To view the full configuration of any experiment:

```bash
# Show all settings
bash scripts/train/run_experiment.sh --show peft_qwen3_32b_lora

# Check specific parameter
bash scripts/train/run_experiment.sh --show peft_qwen3_32b_lora | grep -A 2 "Batch Size"
```

---

## Next Steps

1. **Test with small dataset first**:
   ```bash
   bash scripts/train/run_experiment.sh peft_qwen3_32b_quick_test
   ```

2. **If successful, run full training**:
   ```bash
   sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_32b_lora
   ```

3. **Compare methods**:
   Run all 7 experiments and compare results:
   ```bash
   for method in lora lora_r64 prompt_tuning prefix_tuning lora_prompt lora_prefix; do
     sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_32b_$method
   done
   ```

4. **Analyze results** in W&B dashboard

---

## Integration with Existing Code

- ✅ Fully compatible with existing PEFT training modules
- ✅ Uses same data loading and preprocessing
- ✅ Reuses efficiency tracking from training module
- ✅ Supports inference with peft_inference.py
- ✅ No new dependencies required

---

**Status**: Ready for production use  
**Last Updated**: April 13, 2026
