from datasets import Dataset, concatenate_datasets
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional, Dict
import json
from code.prompt import FACT_CHECK_SYSTEM_PROMPT, FACT_CHECK_USER_PROMPT


class PerturbationType(Enum):
    """Enumeration of perturbation types"""
    NUMERATION_TRUE = "numeration_true"
    NUMERATION_FALSE = "numeration_false"
    APPROXIMATION_TRUE = "approximation_true"
    APPROXIMATION_FALSE = "approximation_false"
    RANGE_TRUE = "range_true"
    RANGE_FALSE = "range_false"
    MASKING_FALSE = "masking_false"
    RANDOM_REPLACEMENT_FALSE = "random_replacement_false"
    ORIGINAL = "original"  # For non-perturbed claims


def load_json_files(file_paths: Union[str, List[str]]) -> List[Dict]:
    """
    Load one or multiple JSON files.
    
    Args:
        file_paths: Single file path or list of file paths
        
    Returns:
        List of all examples from all files
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    all_examples = []
    for file_path in file_paths:
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both single object and array of objects
            if isinstance(data, list):
                all_examples.extend(data)
            else:
                all_examples.append(data)
    
    return all_examples


def expand_perturbations(examples: List[Dict]) -> List[Dict]:
    """
    Expand each example into multiple examples based on available perturbation types.
    Each example will have the claim_id and perturbation_type for tracking.
    
    Args:
        examples: List of raw examples from JSON
        
    Returns:
        Expanded list with one entry per perturbation type
    """
    expanded = []
    
    perturbation_types = [e.value for e in PerturbationType if e != PerturbationType.ORIGINAL]
    
    for example in examples:
        # Add original claim (using normalized_claim)
        original_entry = {
            'claim_id': example['claim_id'],
            'claim': example['normalized_claim'],
            'doc': example['doc'],
            'label': example['label'],
            'perturbation_type': PerturbationType.ORIGINAL.value,
            'url': example.get('url', ''),
            'taxonomy_label': example.get('taxonomy_label', ''),
        }
        expanded.append(original_entry)
        
        # Add perturbed versions if they exist
        for perturb_type in perturbation_types:
            if perturb_type in example and example[perturb_type]:
                # Determine label based on perturbation type
                # *_true perturbations should have the same label as original
                # *_false perturbations should have label "False"
                if perturb_type.endswith('_true'):
                    perturb_label = example['label']
                else:  # ends with '_false'
                    perturb_label = "False"
                
                perturbed_entry = {
                    'claim_id': example['claim_id'],
                    'claim': example[perturb_type],
                    'doc': example['doc'],
                    'label': perturb_label,
                    'perturbation_type': perturb_type,
                    'url': example.get('url', ''),
                    'taxonomy_label': example.get('taxonomy_label', ''),
                }
                expanded.append(perturbed_entry)
    
    return expanded


def convert_to_conversations(
    examples: List[Dict], 
    split: str = "train",
    include_doc: bool = True
) -> List[Dict]:
    """
    Convert examples to conversation format for SFTTrainer.
    
    Args:
        examples: List of expanded examples
        split: "train" or "validation"
        include_doc: Whether to include the evidence document in the prompt
        
    Returns:
        List of examples in conversation format
    """
    messages = []
    
    for example in examples:
        conversations = []
        
        # System message
        conversations.append({
            "role": "system",
            "content": FACT_CHECK_SYSTEM_PROMPT
        })
        
        # User instruction
        conversations.append({
            "role": "user",
            "content": FACT_CHECK_USER_PROMPT
        })
        
        # Construct the claim message with optional evidence
        if include_doc and example['doc']:
            claim_content = f"Claim: {example['claim']}\n\nEvidence: {example['doc']}"
        else:
            claim_content = f"Claim: {example['claim']}"
        
        conversations.append({
            "role": "user",
            "content": claim_content
        })
        
        # Assistant response (only for training)
        if split == "train":
            conversations.append({
                "role": "assistant",
                "content": example['label']
            })
        
        # Build the final entry
        entry = {
            "conversations": conversations,
            "claim_id": example['claim_id'],
            "perturbation_type": example['perturbation_type'],
        }
        
        # For validation, include ground truth
        if split != "train":
            entry["ground_truth"] = example['label']
        
        messages.append(entry)
    
    return messages


# In your data_loader.py (or hf_dataset_instructions.py)

def load_fact_checking_data(
    json_paths: Union[str, List[str]],
    split: str = "train",  # "train", "validation", or "test"
    expand_perturbations_flag: bool = True,
    include_doc: bool = True
) -> Dataset:
    """
    Load fact-checking data from JSON files and convert to conversation format.
    
    Args:
        json_paths: Single JSON file path or list of paths
        split: "train", "validation", or "test"
        expand_perturbations_flag: Whether to expand perturbations into separate examples
        include_doc: Whether to include evidence document in prompts
        
    Returns:
        HuggingFace Dataset ready for SFTTrainer
    """
    # Treat "test" the same as "validation" (no assistant response, include ground_truth)
    effective_split = "validation" if split == "test" else split
    
    # Load raw examples from JSON files
    raw_examples = load_json_files(json_paths)
    
    # Expand perturbations if requested
    if expand_perturbations_flag:
        processed_examples = expand_perturbations(raw_examples)
    else:
        # Just use normalized_claim as-is
        processed_examples = []
        for example in raw_examples:
            processed_examples.append({
                'claim_id': example['claim_id'],
                'claim': example['normalized_claim'],
                'doc': example['doc'],
                'label': example['label'],
                'perturbation_type': PerturbationType.ORIGINAL.value,
                'url': example.get('url', ''),
                'taxonomy_label': example.get('taxonomy_label', ''),
            })
    
    # Convert to conversation format
    conversation_data = convert_to_conversations(
        processed_examples, 
        split=effective_split,
        include_doc=include_doc
    )
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(conversation_data)
    
    return dataset

def load_train_and_validation(
    train_json_paths: Union[str, List[str]],
    val_json_paths: Union[str, List[str]],
    expand_perturbations_flag: bool = True,
    include_doc: bool = True
) -> Dict[str, Dataset]:
    """
    Load both training and validation datasets.
    
    Args:
        train_json_paths: Path(s) to training JSON files
        val_json_paths: Path(s) to validation JSON files
        expand_perturbations_flag: Whether to expand perturbations
        include_doc: Whether to include evidence documents
        
    Returns:
        Dictionary with "train" and "validation" datasets
    """
    train_dataset = load_fact_checking_data(
        train_json_paths,
        split="train",
        expand_perturbations_flag=expand_perturbations_flag,
        include_doc=include_doc
    )
    
    val_dataset = load_fact_checking_data(
        val_json_paths,
        split="validation",
        expand_perturbations_flag=expand_perturbations_flag,
        include_doc=include_doc
    )
    
    return {
        "train": train_dataset,
        "validation": val_dataset
    }


if __name__ == "__main__":
    # Example usage: Load single file
    train_dataset = load_fact_checking_data(
        json_paths="data/train.json",
        split="train",
        expand_perturbations_flag=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"\nDataset features: {train_dataset.features}")
    
    # Print sample conversations
    print("\n" + "="*80)
    print("SAMPLE CONVERSATIONS:")
    print("="*80)
    
    for i in range(min(3, len(train_dataset))):
        print(f"\n{'='*80}")
        print(f"Example {i+1}:")
        print(f"Claim ID: {train_dataset[i]['claim_id']}")
        print(f"Perturbation Type: {train_dataset[i]['perturbation_type']}")
        print(f"\nConversation:")
        for msg in train_dataset[i]['conversations']:
            print(f"\n[{msg['role'].upper()}]")
            print(f"{msg['content'][:200]}..." if len(msg['content']) > 200 else msg['content'])
    
    # Example: Load multiple files with train/val split
    print("\n" + "="*80)
    print("LOADING TRAIN AND VALIDATION:")
    print("="*80)
    
    datasets = load_train_and_validation(
        train_json_paths=["data/train_1.json", "data/train_2.json"],
        val_json_paths="data/val.json",
        expand_perturbations_flag=True
    )
    
    print(f"\nTrain dataset size: {len(datasets['train'])}")
    print(f"Validation dataset size: {len(datasets['validation'])}")
    
    # Show distribution of perturbation types
    print("\n" + "="*80)
    print("PERTURBATION TYPE DISTRIBUTION (Train):")
    print("="*80)
    from collections import Counter
    perturb_counts = Counter(datasets['train']['perturbation_type'])
    for perturb_type, count in perturb_counts.most_common():
        print(f"{perturb_type}: {count}")