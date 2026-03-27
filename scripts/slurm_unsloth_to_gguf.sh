#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=300G
#SBATCH --job-name=unsloth_to_gguf
#SBATCH --output=unsloth_to_gguf.out
#SBATCH --nodelist=gorina9



# Restrict CUDA to GPUs allocated by SLURM
if [ -z "$SLURM_GPUS" ]; then
    # Fallback to SLURM_GPUS_ON_NODE if SLURM_GPUS is not set
    if [ -n "$SLURM_GPUS_ON_NODE" ]; then
        export CUDA_VISIBLE_DEVICES="$SLURM_GPUS_ON_NODE"
        echo "Setting CUDA_VISIBLE_DEVICES to SLURM_GPUS_ON_NODE: $CUDA_VISIBLE_DEVICES"
    fi
else
    export CUDA_VISIBLE_DEVICES="$SLURM_GPUS"
    echo "Setting CUDA_VISIBLE_DEVICES to SLURM_GPUS: $CUDA_VISIBLE_DEVICES"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

uv sync
nvidia-smi

./scripts/setup_llamacpp.sh
pwd
uv run python scripts/demo_unsloth_to_gguf.py