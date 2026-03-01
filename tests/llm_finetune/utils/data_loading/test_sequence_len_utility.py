import pytest
from unittest.mock import Mock, patch

from src.llm_finetune.utils.data_loading.sequence_len_utility import (
    analyze_sequence_lengths,
)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()

    def mock_apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=False
    ):
        return "formatted text"

    def mock_tokenize(text, return_tensors=None, add_special_tokens=True):
        # Simulate tokenization: count words as tokens
        token_count = len(text.split())
        return {"input_ids": list(range(token_count))}

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.return_value = {"input_ids": list(range(10))}
    tokenizer.side_effect = mock_tokenize

    return tokenizer


@pytest.fixture
def mock_dataset():
    """Create a mock HuggingFace dataset."""

    def create_example(idx, seq_len=100, has_evidence=True):
        evidence_text = (
            " Evidence: This is evidence text with multiple tokens for testing"
            if has_evidence
            else ""
        )
        return {
            "claim_id": f"claim_{idx}",
            "perturbation_type": "ORIGINAL",
            "conversations": [
                {"role": "system", "content": "system message"},
                {"role": "assistant", "content": "assistant message"},
                {
                    "role": "user",
                    "content": f"This is claim {idx}{evidence_text}",
                },
            ],
        }

    class MockDataset:
        def __init__(self, size=5):
            self.size = size
            self.data = [create_example(i) for i in range(size)]

        def __len__(self):
            return self.size

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            new_ds = MockDataset(len(indices))
            new_ds.data = [self.data[i] for i in indices]
            return new_ds

    return MockDataset(5)


@pytest.fixture
def mock_tokenizer_with_lengths():
    """Create a tokenizer mock that returns specific lengths."""
    tokenizer = Mock()

    lengths = {
        "formatted text": 150,
        (
            "This is claim 0 Evidence: This is evidence text with "
            "multiple tokens for testing"
        ): 30,
        (
            "This is claim 0 Evidence: This is evidence text with "
            "multiple tokens for testing"
        )
        .split("Evidence:")[1]
        .strip(): 15,
    }

    def mock_apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=False
    ):
        return "formatted text"

    def mock_tokenize(text, return_tensors=None, add_special_tokens=True):
        token_count = lengths.get(text, len(text.split()))
        return {"input_ids": list(range(token_count))}

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.side_effect = mock_tokenize

    return tokenizer


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_basic(
    mock_plt, mock_load_data, mock_dataset, mock_tokenizer_with_lengths
):
    """Test basic functionality of analyze_sequence_lengths."""
    mock_load_data.return_value = mock_dataset

    result = analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="train",
        expand_perturbations=True,
    )

    assert "stats" in result
    assert "sequence_lengths" in result
    assert "too_long_examples" in result
    assert "recommended_max_seq_length" in result
    assert result["stats"]["total_examples"] == 5


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_stats_calculation(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test that statistics are correctly calculated."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": "c1",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                },
                {
                    "claim_id": "c2",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                },
            ]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_load_data.return_value = MockDataset()

    result = analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="train",
    )

    stats = result["stats"]
    assert stats["total_examples"] == 2
    assert stats["min_length"] > 0
    assert stats["max_length"] >= stats["min_length"]
    assert stats["mean_length"] > 0
    assert stats["median_length"] > 0


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_max_examples(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test max_examples parameter limits dataset size."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": f"c{i}",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                }
                for i in range(10)
            ]
            self.selected_count = None

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            self.selected_count = len(indices)
            new_ds = MockDataset()
            new_ds.data = [self.data[i] for i in indices]
            return new_ds

    dataset = MockDataset()
    mock_load_data.return_value = dataset

    result = analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="train",
        max_examples=3,
    )

    assert result["stats"]["total_examples"] == 3


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_too_long_examples(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test detection of sequences exceeding 8192 tokens."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": "long",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                },
            ]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "text"
    mock_tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": list(range(9000))
    }

    mock_load_data.return_value = MockDataset()

    result = analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer,
        split="train",
    )

    assert len(result["too_long_examples"]) == 1
    assert result["too_long_examples"][0]["length"] == 9000


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_recommended_length(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test that recommended_max_seq_length rounds to 1024."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": f"c{i}",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                }
                for i in range(100)
            ]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_load_data.return_value = MockDataset()

    result = analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="train",
    )

    recommended = result["recommended_max_seq_length"]
    assert recommended % 1024 == 0
    assert recommended > 0


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_split_parameter(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test that split parameter is passed to data loader."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": "c1",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                }
            ]

        def __len__(self):
            return 1

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_load_data.return_value = MockDataset()

    analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="validation",
    )

    mock_load_data.assert_called_once()
    call_kwargs = mock_load_data.call_args[1]
    assert call_kwargs["split"] == "validation"


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt")
def test_analyze_sequence_lengths_expand_perturbations_flag(
    mock_plt, mock_load_data, mock_tokenizer_with_lengths
):
    """Test that expand_perturbations flag is passed correctly."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": "c1",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                }
            ]

        def __len__(self):
            return 1

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_load_data.return_value = MockDataset()

    analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        expand_perturbations=False,
    )

    call_kwargs = mock_load_data.call_args[1]
    assert call_kwargs["expand_perturbations_flag"] is False


@patch(
    "src.llm_finetune.utils.data_loading.sequence_len_utility"
    ".load_fact_checking_data"
)
@patch("src.llm_finetune.utils.data_loading.sequence_len_utility.plt.savefig")
def test_analyze_sequence_lengths_saves_plot(
    mock_savefig, mock_load_data, mock_tokenizer_with_lengths
):
    """Test that plot is saved with correct filename."""

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "claim_id": "c1",
                    "perturbation_type": "ORIGINAL",
                    "conversations": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "asst"},
                        {"role": "user", "content": "claim Evidence: doc"},
                    ],
                }
            ]

        def __len__(self):
            return 1

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return self

    mock_load_data.return_value = MockDataset()

    analyze_sequence_lengths(
        json_paths=["test.json"],
        tokenizer=mock_tokenizer_with_lengths,
        split="validation",
    )

    mock_savefig.assert_called_once()
    call_args = mock_savefig.call_args[0]
    assert "sequence_length_analysis_validation.png" in call_args[0]
