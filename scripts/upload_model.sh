#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --job-name=hf_upload_model
#SBATCH --output=scripts/slurm_out/upload_hf/upload_model.out

MODEL_PATH="./models/Qwen3-32B_20260216_115447_numeration_enhanced/epoch-3/merged_model_r32"       # <-- set your model path here
MODEL_NAME="qwen3-32b-numeration-enchanced-batch"       # <-- set your HF repo name here (e.g. "my-llama-finetune")
TYPE="merged"             # <-- set to "merged", "lora" or "checkpoint" if you want to specify the type of model being uploaded (optional)



# Load environment
echo "Activating uv environment..."
#echo $PATH
source .finetune/bin/activate

# ---------------------------------------------------------------------------

CMD="python -m src.llm_finetune.upload_model_hf \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --type $TYPE"

# Print full command (for debugging)
echo "Attempting to upload model to Hugging Face Hub"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Model Upload Completed Successfully!"
echo "=========================================="
echo ""