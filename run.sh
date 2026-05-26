#!/bin/bash
# Usage: bash run.sh <recipe>
#   bash run.sh train-pt-balanced-all
#   bash run.sh infer-pt-all

TRAIN="sbatch scripts/train/slurm_peft_experiment.sh"
INFER="sbatch scripts/inference/slurm_experiment.sh"
# CUDA_VISIBLE_DEVICES=5
case "${1:-help}" in

  # ── Prompt Tuning balanced (oversample True) ──────────────────────────────
  train-pt-balanced-full-data)        $TRAIN peft_qwen3_8b_full_data_pt_balanced ;;
  train-pt-balanced-no-numeration)    $TRAIN peft_qwen3_8b_no_numeration_pt_balanced ;;
  train-pt-balanced-no-approximation) $TRAIN peft_qwen3_8b_no_approximation_pt_balanced ;;
  train-pt-balanced-no-range)         $TRAIN peft_qwen3_8b_no_range_pt_balanced ;;
  train-pt-balanced-no-perturb-split) $TRAIN peft_qwen3_8b_no_perturb_split_pt_balanced ;;
  train-pt-balanced-all)
    sbatch scripts/train/slurm_peft_pt_balanced_all.sh
    ;;

  # ── Prompt Tuning original (unbalanced) ───────────────────────────────────
  train-pt-full-data)        $TRAIN peft_qwen3_8b_prompt_tuning ;;
  train-pt-no-numeration)    $TRAIN peft_qwen3_8b_no_numeration_prompt_tuning ;;
  train-pt-no-approximation) $TRAIN peft_qwen3_8b_no_approximation_prompt_tuning ;;
  train-pt-no-range)         $TRAIN peft_qwen3_8b_no_range_prompt_tuning ;;
  train-pt-no-perturb-split) $TRAIN peft_qwen3_8b_no_perturb_split_prompt_tuning ;;

  # ── LoRA leave-one-out (GGUF export) ──────────────────────────────────────
  train-lora-all)             sbatch scripts/train/slurm_peft_lora_all.sh ;;
  train-lora-full-data)       $TRAIN peft_qwen3_8b_lora_full_data ;;
  train-lora-no-numeration)   $TRAIN peft_qwen3_8b_no_numeration_lora ;;
  train-lora-no-approximation)$TRAIN peft_qwen3_8b_no_approximation_lora ;;
  train-lora-no-range)        $TRAIN peft_qwen3_8b_no_range_lora ;;
  train-lora-no-perturb-split)$TRAIN peft_qwen3_8b_no_perturb_split_lora ;;

  # ── Inference — Prompt Tuning (original, unbalanced) ─────────────────────
  infer-pt-full-data)        $INFER qwen3_8b_pt_full_data ;;
  infer-pt-no-numeration)    $INFER qwen3_8b_pt_no_numeration ;;
  infer-pt-no-approximation) $INFER qwen3_8b_pt_no_approximation ;;
  infer-pt-no-range)         $INFER qwen3_8b_pt_no_range ;;
  infer-pt-no-perturb-split) $INFER qwen3_8b_pt_no_perturb_split ;;
  infer-pt-all)
    $INFER qwen3_8b_pt_full_data
    $INFER qwen3_8b_pt_no_numeration
    $INFER qwen3_8b_pt_no_approximation
    $INFER qwen3_8b_pt_no_range
    $INFER qwen3_8b_pt_no_perturb_split
    ;;

  # ── Inference — Prompt Tuning balanced (oversample True) ─────────────────
  infer-pt-balanced-full-data)        $INFER qwen3_8b_pt_full_data_balanced ;;
  infer-pt-balanced-no-numeration)    $INFER qwen3_8b_pt_no_numeration_balanced ;;
  infer-pt-balanced-no-approximation) $INFER qwen3_8b_pt_no_approximation_balanced ;;
  infer-pt-balanced-no-range)         $INFER qwen3_8b_pt_no_range_balanced ;;
  infer-pt-balanced-no-perturb-split) $INFER qwen3_8b_pt_no_perturb_split_balanced ;;
  infer-pt-balanced-all)
    $INFER qwen3_8b_pt_no_numeration_balanced
    $INFER qwen3_8b_pt_no_approximation_balanced
    $INFER qwen3_8b_pt_no_range_balanced
    $INFER qwen3_8b_pt_no_perturb_split_balanced
    ;;

  # ── Inference — Frontier models (litellm / API-based, no GPU) ────────────
  infer-frontier-claude-opus)    sbatch scripts/inference/slurm_frontier_experiment.sh frontier_claude_opus_47 ;;
  infer-frontier-gpt-55)         sbatch scripts/inference/slurm_frontier_experiment.sh frontier_gpt_55 ;;
  infer-frontier-azure-gpt-54)
    set -a && source .env && set +a
    bash scripts/inference/run_experiment.sh frontier_azure_gpt_54_pro
    ;;
  infer-frontier-all)
    sbatch scripts/inference/slurm_frontier_experiment.sh frontier_claude_opus_47
    sbatch scripts/inference/slurm_frontier_experiment.sh frontier_gpt_55
    ;;

  # ── Help ──────────────────────────────────────────────────────────────────
  help|*)
    echo "Usage: bash run.sh <recipe>"
    echo ""
    echo "Prompt tuning (balanced, oversample True):"
    echo "  train-pt-balanced-all"
    echo "  train-pt-balanced-full-data"
    echo "  train-pt-balanced-no-numeration"
    echo "  train-pt-balanced-no-approximation"
    echo "  train-pt-balanced-no-range"
    echo "  train-pt-balanced-no-perturb-split"
    echo ""
    echo "Prompt tuning (original, unbalanced):"
    echo "  train-pt-full-data / no-numeration / no-approximation / no-range / no-perturb-split"
    echo ""
    echo "LoRA leave-one-out (GGUF):"
    echo "  train-lora-all"
    echo "  train-lora-full-data / no-numeration / no-approximation / no-range / no-perturb-split"
    echo ""
    echo "Inference (prompt tuning, original unbalanced):"
    echo "  infer-pt-all"
    echo "  infer-pt-full-data / no-numeration / no-approximation / no-range / no-perturb-split"
    echo ""
    echo "Inference (prompt tuning, balanced oversample):"
    echo "  infer-pt-balanced-all  (runs 4 ready models; full-data not yet trained)"
    echo "  infer-pt-balanced-no-numeration / no-approximation / no-range / no-perturb-split"
    echo "  infer-pt-balanced-full-data  (requires training first: train-pt-balanced-full-data)"
    echo ""
    echo "Inference (frontier models, API-based):"
    echo "  infer-frontier-all"
    echo "  infer-frontier-claude-opus    (claude-opus-4-7 via Anthropic)"
    echo "  infer-frontier-gpt-55         (gpt-5.5 via OpenAI)"
    echo "  infer-frontier-azure-gpt-54   (azure/gpt-5.4-pro via Azure OpenAI, reads .env)"
    ;;
esac
