#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --job-name=hf_upload_model
#SBATCH --output=scripts/slurm_out/upload_hf/upload_model_%j.out

MODEL_PATH=".models/gguf/Qwen3-8B-baseline_q4_k_m"       # <-- set your model path here
MODEL_NAME="Qwen3-8B-baseline_q4_k_m"       # <-- set your HF repo name here (e.g. "my-llama-finetune")
TYPE="gguf"             # <-- set to "merged", "lora", "checkpoint", or "gguf" if you want to specify the type (optional, auto-detected if empty)
NAMESPACE="iai-group "        # <-- org or username namespace to upload under (optional, defaults to your HF username)
PUBLIC=false        # <-- set to true to make the repo public (default: private)
COMMIT_MESSAGE=""   # <-- custom commit message (optional)


# ---------------------------------------------------------------------------

CMD="uv run python -m src.llm_finetune.upload_model_hf \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME"

# Append optional flags
[[ -n "$TYPE" ]] && CMD="$CMD --type $TYPE"
[[ -n "$NAMESPACE" ]] && CMD="$CMD --namespace $NAMESPACE"
[[ "$PUBLIC" == "true" ]] && CMD="$CMD --public"
[[ -n "$COMMIT_MESSAGE" ]] && CMD="$CMD --commit-message \"$COMMIT_MESSAGE\""

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