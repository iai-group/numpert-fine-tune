"""Fine-tunes FinQA Roberta model for NLI."""

import json
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict


def load_json_data(file_path: str) -> list:
    """Load JSON data from a file and return it as a list of dictionaries."""
    with open(file_path, "r") as file:
        return json.load(file)


def prepare_dataset(train_file: str, test_file: str) -> DatasetDict:
    """Prepare a dataset from train and test JSON files.

    Args:
        train_file: Path to the training data JSON file.
        test_file: Path to the testing data JSON file.

    Returns:
        A DatasetDict object with train and test splits.
    """
    train_data = load_json_data(train_file)
    test_data = load_json_data(test_file)

    # Convert lists of dicts into Dataset objects
    train_dataset = Dataset.from_dict(
        {
            "text": [
                f"{item['claim']} [SEP] {item['doc']}" for item in train_data
            ],
            "label": [item["label"] for item in train_data],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "text": [
                f"{item['claim']} [SEP] {item['doc']}" for item in test_data
            ],
            "label": [item["label"] for item in test_data],
        }
    )

    return DatasetDict(train=train_dataset, test=test_dataset)


def preprocess_data(
    tokenizer: RobertaTokenizer, dataset: Dataset, max_length: int = 512
):
    """Preprocesses the dataset for training.

    Args:
        tokenizer: A tokenizer instance for tokenizing text data.
        dataset: The dataset to preprocess.
        max_length: The maximum token length.

    Returns:
        Tokenized datasets.
    """

    def tokenize_function(example):
        # Tokenize input text
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Ensure tokenized fields are added to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def get_model_and_tokenizer(model_name: str):
    """Initializes the model and tokenizer.

    Args:
        model_name: The name of the pre-trained model.

    Returns:
        A tuple containing the model and tokenizer.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    return model, tokenizer


def fine_tune_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir="./results",
    epochs=3,
    batch_size=8,
):
    """Fine-tunes the model.

    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer for processing inputs.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        output_dir: The directory to save fine-tuned models.
        epochs: The number of epochs to train.
        batch_size: Batch size for training.

    Returns:
        The trained model.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model


def main():
    """Main function for fine-tuning the model."""
    model_name = "iai-group/FinQA_RoBERTa_NLI"
    train_file = "data/raw_data/train_claims_quantemp.json"
    test_file = "data/raw_data/test_claims_quantemp.json"

    # Load and prepare dataset
    dataset = prepare_dataset(train_file, test_file)

    model, tokenizer = get_model_and_tokenizer(model_name)
    tokenized_datasets = preprocess_data(tokenizer, dataset)

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]

    fine_tune_model(model, tokenizer, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
