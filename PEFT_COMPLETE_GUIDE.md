# PEFT Fine-Tuning - Complete Guide

This is your all-in-one guide for using the Parameter-Efficient Fine-Tuning (PEFT) system for numerical fact-checking. Read from top to bottom, or jump to the section you need.

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [What You Can Do](#what-you-can-do)
3. [How to Install](#how-to-install)
4. [Available PEFT Methods](#available-peft-methods)
5. [Training Models](#training-models)
6. [Running Inference](#running-inference)
7. [Comparing Methods](#comparing-methods)
8. [Understanding Results](#understanding-results)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)
11. [File Reference](#file-reference)

---

## Quick Start

Get started in 5 minutes:

```bash
# 1. Setup environment
uv sync

# 2. Test that everything works
uv run bash scripts/test_peft_methods.sh

# 3. Train your first model (LoRA)
uv run python -m src.llm_finetune.peft_finetune \
    --train_json data/perturbed_data/train_split/*.json \
    --val_json data/perturbed_data/val_split/*.json \
    --peft_method lora \
    --num_train_epochs 1

# 4. Run inference on trained model
uv run python -m src.llm_finetune.peft_inference \
    --model_path models/*/checkpoints \
    --test_json data/perturbed_data/test_split/*.json \
    --output_dir results

# 5. Check results
cat results/metrics.json
```

Done! Your first PEFT model is trained and evaluated.

---

## What You Can Do

### Single Method Training
Train one PEFT method and evaluate it:
```bash
uv run python -m src.llm_finetune.peft_finetune --peft_method lora --num_train_epochs 3
```

### Compare Multiple Methods
Train and compare all 5 PEFT methods side-by-side:
```bash
uv run python -m src.llm_finetune.compare_peft_methods \
    --methods lora,prompt_tuning,prefix_tuning,lora_prompt,lora_prefix \
    --num_train_epochs 1
```

### Optimize Hyperparameters
Find the best settings for your task:
```bash
for rank in 8 16 32 64; do
    uv run python -m src.llm_finetune.peft_finetune \
        --peft_method lora --rank $rank
done
```

### Run on SLURM Cluster
Submit jobs to your HPC cluster:
```bash
sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_8b_lora
```

---

## How to Install

### Prerequisites
- Python 3.11+
- GPU with CUDA support (optional, but recommended)
- ~8GB GPU memory minimum

### Setup

```bash
# Install dependencies (already in your project)
uv sync

# Verify installation
uv run python -c "from src.llm_finetune.peft_finetune import PeftMethod; print('✅ Ready!')"
```

That's it! All required packages are already in your project.

---

## Available PEFT Methods

### 1. **LoRA** (Low-Rank Adaptation)
- **Best for**: General purpose, best accuracy-to-efficiency tradeoff
- **Trainable parameters**: ~0.2% of model
- **Training time**: ~30 min per epoch (8B model)
- **Inference latency**: 2-3 ms per sample
- **Use case**: When you want good accuracy with moderate parameter efficiency

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --peft_method lora \
    --rank 32 \
    --num_train_epochs 3
```

**Hyperparameters to tune**:
- `--rank`: 8, 16, 32, 64 (higher = more parameters, potentially better accuracy)

### 2. **Prompt Tuning**
- **Best for**: Extreme parameter efficiency
- **Trainable parameters**: ~0.01% of model
- **Training time**: ~20 min per epoch (fastest)
- **Inference latency**: 2-3 ms per sample
- **Use case**: When you need minimal parameter overhead

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --peft_method prompt_tuning \
    --num_virtual_tokens 20 \
    --num_train_epochs 3
```

**Hyperparameters to tune**:
- `--num_virtual_tokens`: 10, 20, 50, 100 (more tokens = more parameters, potentially better accuracy)

### 3. **Prefix Tuning**
- **Best for**: Layer-wise adaptation with good expressiveness
- **Trainable parameters**: ~0.05% of model
- **Training time**: ~25 min per epoch
- **Inference latency**: 2-3 ms per sample
- **Use case**: When you want expressiveness between Prompt and LoRA

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --peft_method prefix_tuning \
    --num_virtual_tokens 20 \
    --num_train_epochs 3
```

### 4. **LoRA + Prompt Tuning** (Combined)
- **Best for**: Maximum accuracy with moderate efficiency
- **Trainable parameters**: ~0.21% of model
- **Training time**: ~40 min per epoch
- **Inference latency**: 2-3 ms per sample
- **Use case**: When accuracy is more important than parameter efficiency

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --peft_method lora_prompt \
    --rank 32 \
    --num_virtual_tokens 20 \
    --num_train_epochs 3
```

### 5. **LoRA + Prefix Tuning** (Combined)
- **Best for**: Best expressiveness for structured tasks
- **Trainable parameters**: ~0.25% of model
- **Training time**: ~45 min per epoch
- **Inference latency**: 2-3 ms per sample
- **Use case**: Complex tasks requiring maximum model expressiveness

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --peft_method lora_prefix \
    --rank 32 \
    --num_virtual_tokens 20 \
    --num_train_epochs 3
```

---

## Training Models

### Basic Training Command

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --train_json data/perturbed_data/train_split/*.json \
    --val_json data/perturbed_data/val_split/*.json \
    --model "Qwen/Qwen2.5-8B-Instruct" \
    --peft_method lora \
    --rank 32 \
    --num_train_epochs 3 \
    --batch_size 2
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_json` | str | Required | Training data files (supports wildcards) |
| `--val_json` | str | Required | Validation data files |
| `--model` | str | Qwen/Qwen2.5-8B-Instruct | Model identifier (full HuggingFace path) |
| `--peft_method` | str | lora | lora, prompt_tuning, prefix_tuning, lora_prompt, lora_prefix |
| `--rank` | int | 32 | LoRA rank (for LoRA methods) |
| `--num_virtual_tokens` | int | 20 | Virtual tokens for prompt/prefix tuning |
| `--num_train_epochs` | int | 1 | Number of training epochs |
| `--batch_size` | int | 2 | Batch size per device |
| `--grad_accum` | int | 1 | Gradient accumulation steps |
| `--max_seq_length` | int | 16384 | Maximum sequence length |
| `--eval_steps` | float/int | 500 | Evaluation frequency (int=steps, float in (0,1]=fraction) |
| `--use_perturbations` | flag | True | Use perturbed examples |
| `--balance_labels` | flag | False | Balance true/false label distribution |

### Supported Models

You can use any model from HuggingFace Hub. Common choices:

```bash
# Fast (3B parameters)
--model "meta-llama/Llama-3.2-3B-Instruct"

# Balanced (8B parameters, recommended)
--model "Qwen/Qwen2.5-8B-Instruct"

# Powerful (32B parameters)
--model "Qwen/Qwen2.5-32B-Instruct"
```

### What Gets Saved

After training, you'll find:

```
models/[run_name]_[method]/
├── checkpoints/           # The trained model weights
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── training_config.json   # Metadata about this training run
├── dataset_distribution_report.json
└── logs/                  # TensorBoard logs
```

### Monitor Training

All training is logged to Weights & Biases. View in real-time:

```bash
open https://wandb.ai/[your-username]/numpert_finetune_peft
```

---

## Running Inference

### Basic Inference

```bash
uv run python -m src.llm_finetune.peft_inference \
    --model_path models/[run_name]_lora/checkpoints \
    --test_json data/perturbed_data/test_split/*.json \
    --output_dir results
```

### Track Memory and Latency

```bash
uv run python -m src.llm_finetune.peft_inference \
    --model_path models/[run_name]_lora/checkpoints \
    --test_json data/perturbed_data/test_split/*.json \
    --output_dir results \
    --track_memory True
```

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | Required | Path to trained model checkpoints |
| `--test_json` | str | Required | Test data files |
| `--output_dir` | str | ./inference_results | Where to save results |
| `--batch_size` | int | 8 | Batch size for inference |
| `--max_seq_length` | int | 16384 | Maximum sequence length |
| `--track_memory` | bool | False | Track GPU memory usage |

### What Gets Saved

```
results/
├── metrics.json           # Accuracy + efficiency metrics
├── predictions.json       # Detailed predictions
├── predictions.csv        # Easy to analyze
└── inference_config.json  # Metadata
```

### Metrics in metrics.json

**Accuracy Metrics**:
- `accuracy`: Correct predictions / total predictions
- `f1_macro`: Macro-averaged F1 score
- `precision_macro`: Macro-averaged precision
- `recall_macro`: Macro-averaged recall

**Inference Efficiency**:
- `avg_latency_ms`: Average time per sample (milliseconds)
- `min_latency_ms`: Fastest sample
- `max_latency_ms`: Slowest sample
- `avg_throughput_samples_per_sec`: Predictions per second
- `avg_peak_memory_mb`: Average GPU memory usage

---

## Comparing Methods

### Run Full Comparison

Train and evaluate all PEFT methods in one command:

```bash
uv run python -m src.llm_finetune.compare_peft_methods \
    --train_json data/perturbed_data/train_split/*.json \
    --val_json data/perturbed_data/val_split/*.json \
    --test_json data/perturbed_data/test_split/*.json \
    --methods lora,prompt_tuning,prefix_tuning,lora_prompt,lora_prefix \
    --num_train_epochs 1 \
    --batch_size 2
```

### What Gets Saved

```
comparison_results/comparison_YYYYMMDD_HHMMSS/
├── lora/
│   ├── checkpoints/
│   ├── inference_results/
│   └── training_config.json
├── prompt_tuning/
│   └── ...
├── prefix_tuning/
│   └── ...
├── lora_prompt/
│   └── ...
├── lora_prefix/
│   └── ...
└── comparison_report.csv      # Main comparison table
```

### Analysis: comparison_report.csv

Open `comparison_report.csv` to see a table like:

```
Method,Total Params,Trainable Params,Trainable %,Accuracy,F1 (macro),Precision,Recall,Avg Latency (ms),Throughput,Peak Memory (MB),Avg Epoch Time (s),Total Training Time (s)
lora,8400000000,16800000,0.20,0.8945,0.8823,0.9012,0.8634,2.34,427.35,8234.2,32.45,97.35
prompt_tuning,8400000000,840000,0.01,0.8567,0.8401,0.8712,0.8090,2.45,408.16,7856.1,20.12,60.36
prefix_tuning,8400000000,4200000,0.05,0.8756,0.8634,0.8923,0.8345,2.41,414.94,8045.3,26.78,80.34
```

### How to Interpret

**For Parameter Efficiency** (want lower trainable %):
- Prompt Tuning: 0.01% ← Most efficient
- Prefix Tuning: 0.05%
- LoRA: 0.20%

**For Accuracy** (want higher):
- LoRA + Prefix: 0.8923 ← Often best
- LoRA: 0.8945
- Prefix Tuning: 0.8756

**For Speed** (want lower latency, higher throughput):
- Most methods are similar at 2.3-2.5 ms per sample
- Throughput ranges from 400-430 samples/sec

---

## Understanding Results

### Example: Comparing Two Runs

Let's say you train LoRA and Prompt Tuning. Here's how to interpret the results:

**Scenario A: Prompt Tuning wins on efficiency, LoRA wins on accuracy**
```
prompt_tuning: 0.01% params, 85% accuracy, 20 min training
lora:         0.20% params, 89% accuracy, 30 min training
```

**Decision**: If accuracy is critical → use LoRA. If you need fast updates → use Prompt Tuning.

**Scenario B: LoRA+Prefix wins overall**
```
lora_prefix: 0.25% params, 91% accuracy, 45 min training
lora:        0.20% params, 89% accuracy, 30 min training
lora_prompt: 0.21% params, 90% accuracy, 40 min training
```

**Decision**: LoRA+Prefix worth it for 2% accuracy gain if you have time.

### Key Metrics to Focus On

1. **Trainable %** - How efficient is the method?
   - < 0.1% = Extremely efficient
   - 0.1-0.3% = Good efficiency
   - > 0.5% = Less efficient

2. **Accuracy** - How well does it perform?
   - > 90% = Excellent
   - 85-90% = Very good
   - 80-85% = Good
   - < 80% = Needs improvement

3. **Latency (ms)** - How fast?
   - 1-2 ms = Very fast
   - 2-5 ms = Good
   - 5-10 ms = Acceptable
   - > 10 ms = Slow

4. **Training Time** - How long to train?
   - < 30 min = Fast
   - 30-60 min = Normal
   - > 60 min = Slow (or just using more data/epochs)

---

## Advanced Usage

### Hyperparameter Sweep: Finding Optimal LoRA Rank

Find the best rank for your task:

```bash
# Try different ranks
for rank in 8 16 32 64 128; do
    echo "Training LoRA with rank=$rank"
    uv run python -m src.llm_finetune.peft_finetune \
        --train_json data/perturbed_data/train_split/*.json \
        --val_json data/perturbed_data/val_split/*.json \
        --peft_method lora \
        --rank $rank \
        --num_train_epochs 1 \
        --batch_size 2
done

# Then run inference on each and compare
for model_dir in models/*_lora/; do
    uv run python -m src.llm_finetune.peft_inference \
        --model_path "$model_dir/checkpoints" \
        --test_json data/perturbed_data/test_split/*.json \
        --output_dir "results/$(basename $model_dir)"
done
```

### Hyperparameter Sweep: Prompt/Prefix Token Count

Find the best number of virtual tokens:

```bash
for tokens in 10 20 50 100 200; do
    echo "Testing Prompt Tuning with $tokens tokens"
    uv run python -m src.llm_finetune.peft_finetune \
        --train_json data/perturbed_data/train_split/*.json \
        --val_json data/perturbed_data/val_split/*.json \
        --peft_method prompt_tuning \
        --num_virtual_tokens $tokens \
        --num_train_epochs 1
done
```

### Using Different Models

Compare model sizes:

```bash
# Small (3B) - Fast, low memory
uv run python -m src.llm_finetune.peft_finetune \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --peft_method lora

# Medium (8B) - Balanced (recommended)
uv run python -m src.llm_finetune.peft_finetune \
    --model "Qwen/Qwen2.5-8B-Instruct" \
    --peft_method lora

# Large (32B) - Powerful, high memory
uv run python -m src.llm_finetune.peft_finetune \
    --model "Qwen/Qwen2.5-32B-Instruct" \
    --peft_method lora
```

### Memory-Constrained Training

If you run out of GPU memory:

```bash
uv run python -m src.llm_finetune.peft_finetune \
    --train_json data/train/*.json \
    --val_json data/val/*.json \
    --batch_size 1 \
    --grad_accum 4 \
    --max_seq_length 8192
```

This keeps the same effective batch size (1*4) while using less memory per step.

### Monitor with Weights & Biases

All training automatically logs to wandb. View metrics:

```bash
# Set your WandB account
wandb login

# View all runs
open https://wandb.ai/[your-username]/numpert_finetune_peft
```

Key metrics tracked:
- Training/validation loss
- Accuracy metrics
- Training throughput
- GPU memory usage

---

## Troubleshooting

### CUDA Out of Memory Error

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of impact):

1. Reduce batch size:
```bash
--batch_size 1  # Was 2
```

2. Increase gradient accumulation:
```bash
--grad_accum 4  # Was 1
```

3. Reduce max sequence length:
```bash
--max_seq_length 8192  # Was 16384
```

4. Use smaller model:
```bash
--model "meta-llama/Llama-3.2-3B-Instruct"  # Instead of 8B
```

5. For Prompt/Prefix Tuning, reduce virtual tokens:
```bash
--num_virtual_tokens 10  # Was 20
```

### Low Accuracy

**Problem**: Model accuracy is lower than expected

**Possible causes**:

1. Training too short:
```bash
--num_train_epochs 3  # Was 1
```

2. Not enough trainable parameters:
```bash
# For LoRA
--rank 64  # Was 32

# For Prompt/Prefix
--num_virtual_tokens 50  # Was 20
```

3. Model too small:
```bash
--model "Qwen/Qwen2.5-32B-Instruct"  # Larger model
```

4. Verify perturbations enabled:
```bash
--use_perturbations  # Make sure this is set
```

### Model Loading Fails

**Problem**: `ConnectionError` or `OSError` when loading model

**Solutions**:

1. Check internet connection (needs to download model)
```bash
curl https://huggingface.co  # Test connectivity
```

2. Explicitly use a model you've already downloaded:
```bash
--model "Qwen/Qwen2.5-8B-Instruct"
```

3. Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
```

### Missing Data Files

**Problem**: `FileNotFoundError` for data files

**Solutions**:

1. Verify files exist:
```bash
ls data/perturbed_data/train_split/
ls data/perturbed_data/val_split/
```

2. Check wildcard patterns work:
```bash
ls data/perturbed_data/train_split/*.json  # Should show files
```

3. Use explicit paths if wildcards don't work:
```bash
--train_json data/perturbed_data/train_split/file1.json data/perturbed_data/train_split/file2.json
```

### Training Very Slow

**Problem**: Training takes much longer than expected

**Possible causes**:

1. Batch size too small (too many steps):
```bash
--batch_size 4  # Was 2
```

2. Evaluation too frequent:
```bash
--eval_steps 1000  # Was 500
```

3. Sequence length very long:
```bash
--max_seq_length 8192  # Was 16384
```

4. CPU bottleneck - check GPU usage:
```bash
nvidia-smi  # Should show high utilization
```

---

## File Reference

### Core Python Modules

| File | Purpose |
|------|---------|
| `src/llm_finetune/peft_finetune.py` | Main training module (25 KB) |
| `src/llm_finetune/peft_inference.py` | Inference with efficiency tracking (12 KB) |
| `src/llm_finetune/compare_peft_methods.py` | Comparison orchestrator (10 KB) |

### SLURM Scripts

| File | Purpose |
|------|---------|
| `scripts/train/slurm_peft_experiment.sh` | Submit training to SLURM |
| `scripts/inference/slurm_peft_inference.sh` | Submit inference to SLURM |
| `scripts/train/slurm_peft_finetune.sh` | Advanced SLURM training |

### Configuration

| File | Purpose |
|------|---------|
| `scripts/train/experiments.conf` | Experiment definitions |
| `scripts/train/run_experiment.sh` | Experiment launcher |
| `pyproject.toml` | Python dependencies |
| `uv.lock` | Locked versions |

### Data

| Directory | Contents |
|-----------|----------|
| `data/perturbed_data/train_split/` | Training data (10 files) |
| `data/perturbed_data/val_split/` | Validation data (10 files) |
| `data/perturbed_data/test_split/` | Test data (10 files) |

### Output Directories

| Directory | Created by | Contains |
|-----------|-----------|----------|
| `models/` | Training | Trained model checkpoints |
| `results/` | Inference | Predictions and metrics |
| `comparison_results/` | Comparison | Side-by-side comparison |
| `outputs/` | SLURM experiments | SLURM job outputs |

---

## Common Commands Cheatsheet

```bash
# Setup
uv sync

# Test
uv run bash scripts/test_peft_methods.sh

# Train single method
uv run python -m src.llm_finetune.peft_finetune \
    --train_json data/train/*.json \
    --val_json data/val/*.json \
    --peft_method lora \
    --num_train_epochs 3

# Run inference
uv run python -m src.llm_finetune.peft_inference \
    --model_path models/*/checkpoints \
    --test_json data/test/*.json

# Compare all methods
uv run python -m src.llm_finetune.compare_peft_methods \
    --train_json data/train/*.json \
    --val_json data/val/*.json \
    --test_json data/test/*.json \
    --methods lora,prompt_tuning,prefix_tuning,lora_prompt,lora_prefix

# SLURM submission
sbatch scripts/train/slurm_peft_experiment.sh peft_qwen3_8b_lora

# Check SLURM job
squeue -u $(whoami)

# View training in WandB
open https://wandb.ai/[your-username]/numpert_finetune_peft
```

---

## Next Steps

1. **Run Quick Test**: `uv run bash scripts/test_peft_methods.sh` (5 min)
2. **Train Single Method**: Follow section "Training Models" (30 min)
3. **Run Inference**: Follow section "Running Inference" (10 min)
4. **Compare Methods**: Follow section "Comparing Methods" (2-3 hours)
5. **Analyze Results**: Open `comparison_report.csv` (10 min)
6. **Optimize**: Use Advanced Usage section (varies)

---

## Quick FAQ

**Q: Which method should I use?**
A: Start with LoRA. It's the best general-purpose method with good accuracy-efficiency tradeoff.

**Q: How long does training take?**
A: For 8B models: LoRA ~30 min/epoch, Prompt Tuning ~20 min/epoch.

**Q: Can I merge the trained adapter with the base model?**
A: Yes, through the PEFT library. Inference will be much faster but requires code changes.

**Q: Can I use different model sizes?**
A: Yes! Try Llama 3B (fast), Qwen 8B (balanced), or Qwen 32B (powerful).

**Q: Are all dependencies already installed?**
A: Yes! PEFT, transformers, and trl are already in your project.

**Q: Can I run this on CPU?**
A: Technically yes, but it will be very slow. GPU is recommended.

**Q: Where are results saved?**
A: Training → `models/`, Inference → `results/`, Comparison → `comparison_results/`

**Q: How do I monitor training?**
A: All training logs to Weights & Biases (`numpert_finetune_peft` project).

---

**You're ready!** Start with Quick Start section or jump to what you need. Good luck! 🚀
