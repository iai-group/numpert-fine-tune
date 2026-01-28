# Basic fine-tuning with default 1:1 ratio
python unsloth_finetune.py \
  --mode finetune \
  --train_json data/train.json \
  --val_json data/val.json \
  --model qwen-3-8b \
  --max_seq_length 16384

# Fine-tuning with balanced labels and custom ratios
python unsloth_finetune.py \
  --mode finetune \
  --train_json data/train_1.json data/train_2.json \
  --val_json data/val.json \
  --model llama-3.2-3b-instruct \
  --balance_labels \
  --perturb_ratio 1.0 \
  --false_perturb_t_to_f_ratio 0.67 \
  --max_seq_length 20480

# Fine-tuning with claim_id batch matching
python unsloth_finetune.py \
  --mode finetune \
  --train_json data/train.json \
  --val_json data/val.json \
  --batch_match_claim_id \
  --split_large_claims \
  --batch_size 8

# Inference
python unsloth_finetune.py \
  --mode inference \
  --test_json data/test.json \
  --model qwen-3-8b \
  --epoch 1 \
  --use_vllm \
  --max_seq_length 24576