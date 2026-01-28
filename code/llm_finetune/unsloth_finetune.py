import json
from typing import Optional
from sklearn.metrics import f1_score, accuracy_score
import torch
from code.prompt import FACT_CHECK_SYSTEM_PROMPT, FACT_CHECK_USER_PROMPT
#from src.llm_finetune.hf_dataset_instructions import load_fact_checking_data
from code.llm_finetune.config import prefix
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
#from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
#import wandb
import argparse
import random
import os
import time
import re
from tqdm import tqdm
import pandas as pd


def get_model_name_arg(model_name):
    if model_name.startswith("llama-3.2-3b-instruct"):
        model_name_arg = "unsloth/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b-instruct"):
        model_name_arg = "unsloth/Llama-3.2-1B-Instruct"
    elif model_name.startswith("llama-3-8b-instruct"):
        model_name_arg = "unsloth/llama-3-8b-Instruct"
    elif model_name.startswith("llama-3.1-8b-instruct"):
        model_name_arg = "unsloth/Meta-Llama-3.1-8B-Instruct"
    elif model_name.startswith("qwen2.5-1.5b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-1.5B-Instruct"
    elif model_name.startswith("qwen2.5-3b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-3B-Instruct"
    elif model_name.startswith("qwen2.5-7b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-7B-Instruct"
    elif model_name.startswith("qwen2.5-14b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-14B-Instruct"
    elif model_name.startswith("qwen3-4B-instruct"):
        model_name_arg = f"unsloth/Qwen3-4B-Instruct-2507"
    elif model_name.startswith("qwen2.5-32b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-32B-Instruct"
    elif model_name.startswith("Qwen3-8B"):
        model_name_arg = f"unsloth/Qwen3-8B"
    elif model_name.startswith("Qwen3-4B"):
        model_name_arg = f"unsloth/Qwen3-4B"
    elif model_name.startswith("Qwen3-32B"):
        model_name_arg = f"unsloth/Qwen3-32B"
    elif model_name.startswith("Qwen3-235B"):
        model_name_arg = f"unsloth/Qwen3-235B-A22B-Instruct-2507"
    elif model_name.startswith("Qwen3-30B"):
        model_name_arg = f"unsloth/Qwen3-30B-A3B-Instruct-2507"
    return model_name_arg


def finetune(dataset_combo, model_name="", rank=32, epoch=1, use_claim_data=False, use_stance_data=False, use_ranking_data=False):
    max_seq_length =  # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if rank==16:
        rank_postfix = ""
    else:
        rank_postfix = f"_r{rank}"

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    if epoch == 1:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"{prefix}epoch-{epoch-1}/{model_name}{quant_status}/lora_{dataset_combo}{rank_postfix}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_arg, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        fix_tokenizer=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if model_name.startswith("llama"):
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    elif model_name.startswith("qwen"):
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml",
        )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    data = load_fact_checking_data(split="train", sample=False, epoch=epoch, use_claim_data=use_claim_data, use_stance_data=use_stance_data, use_ranking_data=use_ranking_data)
    # data = get_hf_data(dataset, "train", target, "default", 
    #                    cot="cot" in model_name, 
    #                    use_web="web" in model_name)
    data = data.map(formatting_prompts_func, batched = True,)

    # run = wandb.init(
    #     project = "stance",
    #     name = f"{model_name}{quant_status} {dataset_combo} (from epoch {epoch-1} ckp)",
    # )

    if not os.path.exists(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset_combo}{rank_postfix}"):
        os.makedirs(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset_combo}{rank_postfix}")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = data,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 0,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 100,
            learning_rate = 1e-4, # 2e-4
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            # lr_scheduler_type = "linear",
            lr_scheduler_type = "constant",
            seed = 3406+epoch,
            output_dir = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset_combo}{rank_postfix}",
            report_to = "wandb", # Use this for WandB etc
        ),
    )
    if model_name.startswith("llama"):
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif model_name.startswith("qwen"):
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "user\n",
            response_part = "assistant\n",
        )
    trainer.train()
    run.finish()
    
    if not os.path.exists(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/"):
        os.makedirs(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/")
    model.save_pretrained(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset_combo}{rank_postfix}")
    tokenizer.save_pretrained(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset_combo}{rank_postfix}")
    merged_path = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/merged_{dataset_combo}{rank_postfix}"
    
    # Merge LoRA weights into base model and save in 16-bit
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",  # vLLM works best with FP16
    )

def extract_prediction_label_from_vllm(generated_text: str, task: str) -> Optional[str]:
    # valid_labels = ['PARTIALLY REFUTES', 'PARTIALLY SUPPORTS', 'SUPPORTS', 'REFUTES', 'No', 'Yes', "0", "1"]
    generated_text = generated_text.strip().rstrip().replace("\n", "\t").replace(".", "")
    
    print(generated_text)
    if "</think>" not in generated_text:
        if "\t" in generated_text:
            generated_text = generated_text.split("\t")[0].strip().rstrip()
        if generated_text in valid_labels[task]:
            return generated_text, True
        else:
            try:
                label_json = json.loads(generated_text) # try to parse as json
                return label_json.get("label", random.choice(valid_labels[task])), label_json.get("label", None) in valid_labels[task]
            except Exception as e:
                print(e)
            # Fallback: return a random valid label if extraction fails
                return random.choice(valid_labels[task]), False
    else:
        label = generated_text.split("</think>")[1].strip().rstrip()
        if label in valid_labels[task]:
            return label, True
        else:
            # Fallback: return a random valid label if extraction fails
            return random.choice(valid_labels[task]), False

def extract_prediction_label(generated_text: str) -> Optional[str]:
    """
    Extract the prediction label from generated text in the format:
    <think>...</think>
    LABEL<|im_end|>
    
    Args:
        generated_text: The full generated text from the model
        
    Returns:
        The extracted label or None if not found
    """
    # Valid labels for this task
    valid_labels = ['PARTIALLY REFUTES', 'PARTIALLY SUPPORTS', 'SUPPORTS', 'REFUTES', 'No', 'Yes', "0", "1"]
    
    # Pattern to match the label after </think> and before <|im_end|>
    pattern = r'</think>\s*\n\s*([A-Z\s]+)<\|im_end\|>'
    
    match = re.search(pattern, generated_text)
    if match:
        extracted_label = match.group(1).strip()
        # Validate that it's one of the expected labels
        if extracted_label in valid_labels:
            return extracted_label
    
    # Fallback: look for any of the valid labels at the end of the text
    for label in valid_labels:
        if label in generated_text:
            # Check if it's near the end of the text
            last_occurrence = generated_text.rfind(label)
            if last_occurrence > len(generated_text) - 100:  # Within last 100 chars
                return label
    
    return None

def inference(dataset_combo, modified_stance, model_name="llama-3.2-1b-instruct", rank=16, epoch=0, use_vllm=False, use_claim_data=False, use_stance_data=False, use_ranking_data=False, use_icl=False):

    if rank==16:
        rank_postfix = ""
    else:
        rank_postfix = f"_r{rank}"

    # 5120 due to long wiki/web info repetition in fewshot
    max_seq_length = 10240 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    if epoch != 0:
        model_name_arg = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset_combo}{rank_postfix}"
    else:
        model_name_arg = get_model_name_arg(model_name)

    if use_vllm and epoch == 0:
        llm = LLM(model=(model_name_arg + "-bnb-4bit").lower(), max_model_len=max_seq_length, dtype="bfloat16", 
                  gpu_memory_utilization=0.7)
        tok = AutoTokenizer.from_pretrained(model_name_arg + "-bnb-4bit", trust_remote_code=True)
        print(tok.chat_template)
    elif use_vllm and epoch != 0:
        llm = LLM(model=(model_name_arg), max_model_len=max_seq_length)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_arg, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            fix_tokenizer=False
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    
    
    cot = " data = get_averitec_data()cot" in model_name
    data = load_fact_checking_data(split="test", sample=False, epoch=epoch, use_claim_data=use_claim_data, use_stance_data=use_stance_data, use_ranking_data=use_ranking_data, use_fewshot_icl=use_icl)
    predictions = []
    raw_predictions = []
    ground_truth = data["ground_truth"]
    task_ground_truth = {"claim_detection": [], "stance_detection": [], "ranking": []}
    task_predictions = {"claim_detection": [], "stance_detection": [], "ranking": []}

    answers_ls = []
    if "cot" in model_name:
        chains_of_thought = []
        # if "fewshot" not in model_name:
            # conclusions = []
    vllm_postfix = "-vllm" if use_vllm else ""
    invalid = 0
    os.makedirs(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}/", exist_ok=True)
    with open(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}/raw_output.txt", "w") as raw_output:
        n_candidates = 1
        if use_vllm:
            sampling_params = SamplingParams(temperature=0, max_tokens=1024, seed=42)
            prompt = tok.apply_chat_template(
                data["conversations"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # strip any reasoning tags if present
            # prompt = prompt.replace("<|thinking|>", "").replace("<|/thinking|>", "")
            outputs = llm.generate(prompt, sampling_params)
            # FIX: Extract strings from nested list structure
            for i, output in enumerate(outputs):
                answer = output.outputs[0].text  # Get the actual text string
                task = data[i]["task"]
                gt = ground_truth[i]
                
                # Extract label from the text string
                # print(answer)
                label, valid = extract_prediction_label_from_vllm(answer, task)
                if not valid:
                    invalid += 1
                answers_ls.append(answer)  # Append string, not list
                predictions.append(label)
                task_ground_truth[task].append(gt)
                task_predictions[task].append(label)
                
                # Write to file
                raw_output.write(str(data[i]["conversations"]) + "\n")
                raw_output.write(answer + "\n")
                raw_output.write(f"Extracted: {label}\n\n")
                
                # print(f"{task}: GT={gt}, Pred={label}")
        else:
            for i in tqdm(range(len(data))):
                task = data[i]["task"]
                task_ground_truth[task].append(ground_truth[i])
                raw_output.write(str(data[i]["conversations"]) + "\n")
                # if i > 0: # for debugging
                model_inputs = tokenizer.apply_chat_template(
                    data[i]["conversations"], 
                    tokenize = True,
                    add_generation_prompt = True, # Must add for generation
                    return_tensors = "pt",
                )
                # Normalize to dict if tokenizer returns a single Tensor
                if isinstance(model_inputs, torch.Tensor):
                    model_inputs = {"input_ids": model_inputs}

                # Move tensors to cuda
                model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
                
                if "attention_mask" not in model_inputs:
                    pad_id = getattr(tokenizer, "pad_token_id", None)
                    if pad_id is None:
                        # fallback: create mask assuming non-zero tokens are real tokens
                        model_inputs["attention_mask"] = (model_inputs["input_ids"] != 0).long()
                    else:
                        model_inputs["attention_mask"] = (model_inputs["input_ids"] != pad_id).long()

                temperature = 0.001
                outputs = model.generate(**model_inputs, max_new_tokens = 1024,
                            use_cache = False, temperature = temperature,
                            num_return_sequences=n_candidates)
                
                
                
                decoded_outputs = tokenizer.batch_decode(outputs)
                answers = [output for output in decoded_outputs]
                labels = [extract_prediction_label(output) for output in decoded_outputs]
                task_predictions[task].append(labels[0])
                raw_output.write(str(decoded_outputs) + "\n")
                answers_ls.append(answers)
                predictions.append(labels[0])
                # print(task, ground_truth[i], labels[0])
            data = data.add_column("predictions", predictions)

    predictions_json_path = f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}/predictions.json"
    data.to_json(predictions_json_path)
    print(f"Invalid predictions (vllm extraction failures): {invalid} out of {len(data)}")
    # Collect metrics for each task
    metrics = {
        "Task": [],
        "F1 Macro": [],
        "F1 Weighted": [],
        "Accuracy": [],
    }
    # Also collect per-label F1s for each task
    per_label_f1 = {}

    for task in task_ground_truth.keys():
        gt = task_ground_truth[task]
        pred = task_predictions[task]
        f1_macro = f1_score(gt, pred, average="macro", zero_division=0)
        f1_weighted = f1_score(gt, pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(gt, pred)
        metrics["Task"].append(task)
        metrics["F1 Macro"].append(f1_macro)
        metrics["F1 Weighted"].append(f1_weighted)
        metrics["Accuracy"].append(accuracy)

        # Compute per-label F1 score
        unique_labels = set(gt) | set(pred)
        per_label_f1[task] = {}
        for label in unique_labels:
            f1 = f1_score(gt, pred, labels=[label], average="macro", zero_division=0)
            per_label_f1[task][label] = f1

    metrics_df = pd.DataFrame(metrics)
    # Write metrics to a LaTeX file with model name in the filename
    metrics_dir = f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    metrics_file = os.path.join(metrics_dir, f"metrics_{model_name}.tex")
    with open(metrics_file, "w") as f:
        f.write(metrics_df.to_latex(index=False, float_format="%.3f"))
    print(metrics_df.to_latex(index=False, float_format="%.3f"))

    # Optionally, print per-label F1s as well
    for task, label_f1s in per_label_f1.items():
        print(f"Per-label F1 for task '{task}':")
        for label, f1 in label_f1s.items():
            print(f"  {label}: {f1:.3f}")

    # Also write per-label F1s to the LaTeX file
    with open(metrics_file, "a") as f:
        f.write("\n% Per-label F1 scores\n")
        for task, label_f1s in per_label_f1.items():
            f.write(f"\\textbf{{Per-label F1 for task '{task}':}}\n")
            f.write("\\begin{itemize}\n")
            for label, f1 in label_f1s.items():
                f.write(f"  \\item {label}: {f1:.3f}\n")
            f.write("\\end{itemize}\n")
    
    results_df = pd.DataFrame(answers_ls)
    vllm_postfix = "-vllm" if use_vllm else ""

    if epoch != 0:
        if not os.path.exists(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}"):
            os.makedirs(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}")
        results_df.to_csv(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}/{target}_{modified_stance}.csv", index=False)
    else:
        if not os.path.exists(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}"):
            os.makedirs(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}")
        results_df.to_csv(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}/{target}_{modified_stance}.csv", index=False)


def inference_wiki(modified_stance, model_name="llama-3.2-1b-instruct", sentiment=False, use_vllm=True):

    max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    model_name_arg = get_model_name_arg(model_name)

    if use_vllm:
        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
        llm = LLM(model=(model_name_arg + "-bnb-4bit").lower(), max_model_len=max_seq_length, dtype="bfloat16", 
                  device="cuda", gpu_memory_utilization=0.7,
                  disable_cuda_graph=True)   
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_arg, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    data = get_hf_wiki_data(modified_stance, use_web="web" in model_name, sentiment=sentiment)
    predictions = []
    raw_predictions = []
    answers_ls = []

    n_candidates = 1
    if use_vllm:
        outputs = llm.chat(data["conversations"], sampling_params)
        answers_ls = [[output.outputs[0].text] for output in outputs]
    else:
        for i in tqdm(range(len(data))):
            # if i > 0: # for debugging
                # break
            inputs = tokenizer.apply_chat_template(
                data[i]["conversations"], 
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            temperature = 0.001
            outputs = model.generate(input_ids = inputs, max_new_tokens = 1024,
                        use_cache = True, temperature = temperature,
                        num_return_sequences=n_candidates)
            
            if model_name.startswith("llama"):
                llama_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_llama_outputs(output, cot=False) for output in llama_outputs]
            elif model_name.startswith("qwen"):
                qwen_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_qwen_outputs(output, cot=False) for output in qwen_outputs]
            answers_ls.append(answers)

    datasets = ["cstance", "pstance2", "semeval"]
    targets_ls = [
        ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
        ["bernie", "trump", "biden"],
        ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"]
    ]

    results_df = {"dataset": [], "target": []}
    for dataset, targets in zip(datasets, targets_ls):
        for target in targets:
            results_df["dataset"].append(dataset)
            results_df["target"].append(target)

    for i, answers in enumerate(answers_ls):
        
        raw_predictions.append(answers)
        # eliminate non-alphabetical characters
        answers = ["".join(filter(str.isalpha, ans)) for ans in answers]
        sample_predictions = []
        if sentiment:
            for ans in answers:
                if ans.lower() in ["positive"]:
                    sample_predictions.append("POSITIVE")
                elif ans.lower() == "negative":
                    sample_predictions.append("NEGATIVE")
                elif ans.lower() in ["none", "neutral"] and results_df["dataset"][i] != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")            
        else:
            for ans in answers:
                if ans.lower() in ["favor", "favour", "favorable", "favourable"]:
                    sample_predictions.append("FAVOR")
                elif ans.lower() == "against":
                    sample_predictions.append("AGAINST")
                elif ans.lower() in ["none", "neutral"] and results_df["dataset"][i] != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")
        predictions.append(sample_predictions)

    for i in range(n_candidates):
        results_df[f"prediction_{i}"] = [pred[i] for pred in predictions]
    for i in range(n_candidates):
        results_df[f"raw_prediction_{i}"] = [raw_pred[i] for raw_pred in raw_predictions]
    results_df = pd.DataFrame(results_df) 

    vllm_postfix = "-vllm" if use_vllm else ""
    char_dir = "sentiment" if sentiment else "stance"
    file_name = "web" if "web" in model_name else "wiki"
    stance_postfix = "" if modified_stance == "default" else f"_{modified_stance}"

    if not os.path.exists(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}"):
        os.makedirs(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}")
    results_df.to_csv(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}/{file_name}{stance_postfix}.csv", index=False)


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", help="Mode: local, finetune, inference, inference_wiki_stance, inference_wiki_sentiment")
    parser.add_argument("--dataset", default="cstance", help="Dataset: cstance, pstance2, semeval")
    parser.add_argument("--target", default="face_masks", help="Target: face_masks, fauci, school_closures, stay_at_home_orders, bernie, trump, biden, atheism, climate_change_is_a_real_concern, feminist_movement, hillary_clinton, legalization_of_abortion")
    parser.add_argument("--model", default="llama-3.2-1b-instruct", 
                        help="Model: llama and qwen models")
    parser.add_argument("--stance", default="default",
                        help="Stance: none, default, neutral, favour, against")
    parser.add_argument("--epoch", default=1, type=int,
                        help="Epoch: 1, 2, 3 for finetuning, 0 for inference without finetuning")
    parser.add_argument("--rank", default=32, type=int,
                        help="Rank: 16, 32 for finetuning")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for inference", default=False)
    parser.add_argument("--use_claim_data", action="store_true",
                        help="Use claim data for training", default=False)
    parser.add_argument("--use_stance_data", action="store_true",
                        help="Use Stance data for training", default=False)
    parser.add_argument("--use_ranking_data", action="store_true",
                        help="Use Ranking data for training", default=False)
    parser.add_argument("--icl", action="store_true",
                        help="Use in-context learning for inference without finetuning", default=False)
    args = parser.parse_args()

    mode = args.mode
    dataset = args.dataset
    target = args.target
    model = args.model
    stance = args.stance
    epoch = args.epoch
    rank = args.rank
    use_vllm = args.use_vllm
    use_claim_data = args.use_claim_data
    use_stance_data = args.use_stance_data
    use_ranking_data = args.use_ranking_data
    use_icl = args.icl

    datasets = ["cstance", "pstance2", "semeval"]
    targets_ls = [
        ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
        ["bernie", "trump", "biden"],
        ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"],
    ]

    # inference wiki
    if mode == "inference_wiki_stance":
        print(f"Running wiki stance inference for {stance} - {model}")
        inference_wiki(stance, model, sentiment=False, use_vllm=use_vllm)
    if mode == "inference_wiki_sentiment":
        print(f"Running wiki sentiment inference for {stance} - {model}")
        inference_wiki(stance, model, sentiment=True, use_vllm=use_vllm)

    # finetuning
    if mode == "finetune":
        print(f"Epoch {epoch}!")
        print(f"Running finetuning for {dataset} - {target} - {model} - rank {rank} - epoch {epoch}")
        finetune(f"{dataset}_{target}", model, rank=rank, epoch=epoch, use_claim_data=use_claim_data, use_stance_data=use_stance_data, use_ranking_data=use_ranking_data)

    # inference
    if mode == "inference":
        print(f"Epoch {epoch}!")
        print(f"Running inference for {dataset} - {model} - rank {rank} - epoch {epoch}")
        # for target in targets_ls[datasets.index(dataset)]:
        inference(f"{dataset}_{target}", "none", model, rank=rank, epoch=epoch, use_vllm=use_vllm, use_claim_data=use_claim_data, use_stance_data=use_stance_data, use_ranking_data=use_ranking_data, use_icl=use_icl)
        # time.sleep(4) # avoid GPU overload
            # inference(f"{dataset}_{target}", "default", model, rank=rank, epoch=epoch, use_vllm=False)
            # time.sleep(4) # avoid GPU overload

    # inference without finetuning
    if mode == "local":
        print(f"Running inference without finetuning for {dataset} - {target} - {stance} - {model}")
        inference(f"{dataset}_{target}", stance, model, epoch=0, use_vllm=False)