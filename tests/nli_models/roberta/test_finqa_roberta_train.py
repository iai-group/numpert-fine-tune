"""Tests for training the FinQA RoBERTa model."""

import pytest
import json
from unittest.mock import patch, MagicMock

import torch
from code.nli_models.roberta.finqa_roberta import (
    configure_wandb,
    evaluate_model,
    fine_tune_model,
    get_model_and_tokenizer,
)
from code.nli_models.utils.data_loader import (
    load_json_data,
    prepare_dataset,
    preprocess_data,
)
from datasets import Dataset, DatasetDict
import os

# Test data for mocking
MOCK_TRAIN_DATA = [
    {"claim": "Claim 1", "doc": "Document 1", "label": "True"},
    {"claim": "Claim 2", "doc": "Document 2", "label": "False"},
]
MOCK_TEST_DATA = [
    {"claim": "Claim 3", "doc": "Document 3", "label": "False"},
    {"claim": "Claim 4", "doc": "Document 4", "label": "True"},
]
MOCK_VAL_DATA = [
    {"claim": "Claim 5", "doc": "Document 5", "label": "False"},
    {"claim": "Claim 6", "doc": "Document 6", "label": "True"},
]


@pytest.fixture
def mock_json_files(tmp_path):
    """Fixture to create temporary JSON files for train and test data."""
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    val_path = tmp_path / "val.json"

    with open(train_path, "w") as f:
        json.dump(MOCK_TRAIN_DATA, f)
    with open(test_path, "w") as f:
        json.dump(MOCK_TEST_DATA, f)
    with open(val_path, "w") as f:
        json.dump(MOCK_VAL_DATA, f)

    return train_path, test_path, val_path


def test_load_json_data(mock_json_files):
    """Test that JSON data is loaded correctly."""
    train_path, _, _ = mock_json_files
    data = load_json_data(train_path)
    assert isinstance(data, list), "Loaded data should be a list."
    assert len(data) == len(MOCK_TRAIN_DATA), "Loaded data length mismatch."
    assert (
        data[0]["claim"] == MOCK_TRAIN_DATA[0]["claim"]
    ), "Claims do not match."


@patch("code.nli_models.utils.data_loader.Dataset.map")
def test_preprocess_data(mock_map):
    """Test that the dataset is tokenized correctly."""
    # Mock tokenizer
    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {
        "input_ids": [101, 2009, 2001, 1037, 2204, 2154, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
    }

    # Mock dataset and mapping behavior
    dataset = Dataset.from_dict(
        {
            "text": ["Example text"],
            "label": [1],
        }
    )
    mock_map.side_effect = lambda func, batched: Dataset.from_dict(
        {
            "text": ["Example text"],
            "label": [1],
            "input_ids": [[101, 2009, 2001, 1037, 2204, 2154, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
        }
    )

    tokenized_dataset = preprocess_data(tokenizer_mock, dataset)
    mock_map.assert_called_once()
    assert (
        "input_ids" in tokenized_dataset.features
    ), "Tokenized dataset missing 'input_ids'."
    assert (
        "attention_mask" in tokenized_dataset.features
    ), "Tokenized dataset missing 'attention_mask'."


@patch("code.nli_models.utils.data_loader.Dataset")
def test_prepare_dataset(mock_dataset, mock_json_files):
    """Test that the dataset is prepared correctly."""
    mock_dataset.from_dict.side_effect = lambda data: Dataset.from_dict(data)

    train_path, test_path, val_path = mock_json_files
    dataset_dict, label_map = prepare_dataset(train_path, test_path, val_path)

    assert isinstance(
        dataset_dict, DatasetDict
    ), "Prepared dataset should be a DatasetDict."
    assert isinstance(label_map, dict), "Label map should be a dictionary."
    assert (
        "train" in dataset_dict and "test" in dataset_dict
    ), "Dataset splits missing."
    assert len(dataset_dict["train"]) == len(
        MOCK_TRAIN_DATA
    ), "Train dataset length mismatch."
    assert len(dataset_dict["test"]) == len(
        MOCK_TEST_DATA
    ), "Test dataset length mismatch."
    assert len(dataset_dict["val"]) == len(
        MOCK_VAL_DATA
    ), "Val dataset length mismatch."


@patch("code.nli_models.roberta.finqa_roberta.AutoTokenizer.from_pretrained")
@patch(
    "code.nli_models.roberta.finqa_roberta.AutoModelForSequenceClassification.from_pretrained"  # noqa E501
)
def test_get_model_and_tokenizer(mock_model, mock_tokenizer):
    """Test that the model and tokenizer are loaded correctly."""
    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    model_name = "iai-group/FinQA_RoBERTa_NLI"
    label_map = {"True": 0, "False": 1}
    dropout_rate = 0.1
    model, tokenizer = get_model_and_tokenizer(
        model_name, label_map=label_map, dropout_rate=0.1
    )

    mock_model.assert_called_once_with(
        model_name,
        num_labels=len(label_map),
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate,
    )
    mock_tokenizer.assert_called_once_with(model_name)
    assert model is not None, "Model should not be None."
    assert tokenizer is not None, "Tokenizer should not be None."


@patch("code.nli_models.roberta.finqa_roberta.Trainer")
def test_fine_tune_model(mock_trainer):
    """Test the fine-tuning process."""
    mock_trainer.return_value = MagicMock(train=MagicMock())
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    train_dataset = MagicMock()
    val_dataset = MagicMock()

    model = fine_tune_model(
        mock_model,
        mock_tokenizer,
        train_dataset,
        val_dataset,
        output_dir="./temp_results",
        epochs=5,
        batch_size=8,
    )

    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    assert model is not None, "Trained model should not be None."


@patch.dict("os.environ", {}, clear=True)
@patch("code.nli_models.roberta.finqa_roberta.wandb.init")
def test_configure_wandb(mock_wandb):
    """Test wandb configuration."""
    configure_wandb()

    mock_wandb.assert_called_once_with(
        project="numpert",
        entity="iai-group",
        name="finqa_nli_finetuning",
        config={"model_name": "iai-group/FinQA_RoBERTa_NLI"},
    )
    assert (
        "WANDB_IGNORE_GLOBS" in os.environ
    ), "WANDB_IGNORE_GLOBS should be set in os.environ."


def test_evaluate_model():
    """Test model evaluation."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = MagicMock(
        logits=torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    )

    test_dataset = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
        "labels": [0, 1],
    }
    metrics = evaluate_model(mock_model, test_dataset)

    assert "accuracy" in metrics, "Metrics should include accuracy."
    assert "f1_weighted" in metrics, "Metrics should include f1_weighted."
