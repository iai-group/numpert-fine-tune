#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --job-name=conda_setup
#SBATCH --output=conda_setup.out
# Activate the user environment (uenv)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uenv verbose cuda-12.4.0 cudnn-12.x-8.8.0
uenv miniconda3-py312

# Create and activate the Conda environment
conda create -n numpert-finetune -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.4 -c nvidia -y
conda activate numpert-finetune


# Install Python packages
pip install -r requirements.txt