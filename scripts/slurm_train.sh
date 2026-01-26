#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --mem=400G
#SBATCH --job-name=numpert_3classes
#SBATCH --output=numpert_3classes.out
 
# Activate environment
uenv verbose cuda-12.4.0 cudnn-12.x-8.8.0
uenv miniconda3-py311
conda activate transformer_cuda124
conda env list
python -m code.nli_models.roberta.finqa_roberta