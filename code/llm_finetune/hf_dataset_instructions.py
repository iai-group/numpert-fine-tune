from datasets import Dataset, concatenate_datasets, load_dataset
from enum import Enum
from code.prompt import FACT_CHECK_SYSTEM_PROMPT, FACT_CHECK_USER_PROMPT


def load_numpert_dataset(hf_dataset_name):
    stance_dataset = load_dataset(hf_dataset_name)
    return stance_dataset

def convert_data_to_conversations(dataset: Dataset, split: str = "train") -> Dataset:
    # Convert the dataset to a conversation format
    messages = []
    for example in dataset:
        conversations = []
        content = FACT_CHECK_SYSTEM_PROMPT
        conversations.append({
            "role": "system",
            "content": content
        })
        # Add system message once at the beginning
        conversations.append({
            "role": "user",
            "content": FACT_CHECK_USER_PROMPT
        })
        conversations.append({
            "role": "user",
            "content": example["text"]
        })
        if split == "train":
            conversations.append({
                "role": "assistant",
                "content": str(example["label"])
            })
        
        if split == "train":
            messages.append({"conversations": conversations})
        else:
            messages.append({"conversations": conversations, "ground_truth": str(example["label"])})
    return Dataset.from_list(messages)

def load_fact_checking_data(numpert_dataset: str = "hf_numpert_url", split="train") -> Dataset:
    numpert_dataset = load_numpert_dataset(numpert_dataset)[split]
    numpert_dataset = numpert_dataset.rename_column("class_label", "label")
    numpert_dataset = numpert_dataset.rename_column("Text", "text")

    #datasets = []


    #final_dataset = concatenate_datasets(datasets)
    return numpert_dataset

if __name__ == "__main__":
    final_dataset = load_fact_checking_data()
    print(final_dataset)
    print("\nSample conversations:")
    for i in range(min(5, len(final_dataset))):
        print(f"\nExample {i+1}:")
        print(f"Role: {final_dataset[i]['role']}")
        print(f"Content: {final_dataset[i]['content']}")