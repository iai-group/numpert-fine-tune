#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=100G
#SBATCH --nodelist=gorina9
#SBATCH --job-name=peft_sample
#SBATCH --output=scripts/slurm_out/peft/peft_sample_%j.out

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$REPO_ROOT"

uv run python -c "
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json

model_path = 'models/Qwen-Qwen3-8B-jolly-falcon-calculates_prompt_tuning/checkpoints'
test_file  = 'data/perturbed_data/test_split/approximation_False.json'
N_SAMPLES  = 5

print('Loading model...')
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path, device_map='auto',
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

with open(test_file) as f:
    samples = json.load(f)[:N_SAMPLES]

for i, ex in enumerate(samples):
    messages = ex.get('conversations', [])
    if isinstance(messages, str):
        import ast; messages = ast.literal_eval(messages)
    messages = [m for m in messages if m['role'] in ('system', 'user')]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    label = ex.get('label', ex.get('conversations', [{}])[-1].get('content', '?') if isinstance(ex.get('conversations'), list) else '?')

    print(f'--- Sample {i+1} ---')
    print(f'Label : {label}')
    print(f'Output: {response[:500]}')
    print()
"
