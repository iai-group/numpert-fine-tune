from __future__ import annotations
from unittest.mock import patch
import pytest

"""Tests for conversation.py module."""


from src.llm_finetune.data_loading_utils.conversation import (
    convert_to_conversations,
)


@pytest.fixture
def mock_prompts():
    """Fixture to mock prompt constants."""
    return {
        "system": "You are a fact-checking assistant.",
        "user": "Claim: {claim}\nEvidence: {evidence}",
    }


@pytest.fixture
def sample_examples():
    """Fixture providing sample expanded examples."""
    return [
        {
            "claim_id": "claim_001",
            "claim": "The Earth is round.",
            "doc": "Scientific evidence confirms the Earth is spherical.",
            "label": "True",
            "perturbation_type": "ORIGINAL",
        },
        {
            "claim_id": "claim_002",
            "claim": "Water boils at 100 degrees.",
            "doc": "At sea level, water boils at 100°C.",
            "label": "False",
            "perturbation_type": "APPROXIMATION_FALSE",
        },
    ]


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_basic(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test basic conversion to conversation format."""
    mock_system_prompt.__str__.return_value = "System prompt"
    mock_system_prompt = "You are a fact-checker."
    mock_user_prompt = "Claim: {claim}\nEvidence: {evidence}"

    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "You are a fact-checker.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Claim: {claim}\nEvidence: {evidence}",
    ):
        result = convert_to_conversations(sample_examples)

    assert len(result) == 2
    assert all("conversations" in entry for entry in result)
    assert all("claim_id" in entry for entry in result)
    assert all("perturbation_type" in entry for entry in result)
    assert all("label" in entry for entry in result)


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_structure(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test conversation structure has correct roles and format."""
    mock_system_prompt = "System."
    mock_user_prompt = "Claim: {claim}\nEvidence: {evidence}"

    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        mock_system_prompt,
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        mock_user_prompt,
    ):
        result = convert_to_conversations([sample_examples[0]])

    conv = result[0]["conversations"]
    assert len(conv) == 3
    assert conv[0]["role"] == "system"
    assert conv[1]["role"] == "user"
    assert conv[2]["role"] == "assistant"


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_label_true(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test label conversion for True case."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nEvidence: {evidence}",
    ):
        result = convert_to_conversations([sample_examples[0]])

    assert result[0]["label"] == '{"label": true}'
    assert result[0]["conversations"][2]["content"] == '{"label": true}'


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_label_false(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test label conversion for False case."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nEvidence: {evidence}",
    ):
        result = convert_to_conversations([sample_examples[1]])

    assert result[0]["label"] == '{"label": false}'
    assert result[0]["conversations"][2]["content"] == '{"label": false}'


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_include_doc_true(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test evidence is included when include_doc=True."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nE: {evidence}",
    ):
        result = convert_to_conversations(
            [sample_examples[0]],
            include_doc=True,
        )

    user_content = result[0]["conversations"][1]["content"]
    assert sample_examples[0]["doc"] in user_content


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_include_doc_false(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test evidence is excluded when include_doc=False."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nE: {evidence}",
    ):
        result = convert_to_conversations(
            [sample_examples[0]],
            include_doc=False,
        )

    user_content = result[0]["conversations"][1]["content"]
    assert sample_examples[0]["doc"] not in user_content


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_preserves_metadata(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test claim_id and perturbation_type are preserved."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nE: {evidence}",
    ):
        result = convert_to_conversations(sample_examples)

    for i, output_entry in enumerate(result):
        assert output_entry["claim_id"] == sample_examples[i]["claim_id"]
        assert (
            output_entry["perturbation_type"]
            == sample_examples[i]["perturbation_type"]
        )


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_empty_input(
    mock_user_prompt,
    mock_system_prompt,
):
    """Test handling of empty examples list."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nE: {evidence}",
    ):
        result = convert_to_conversations([])

    assert result == []


@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT"
)
@patch(
    "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT"
)
def test_convert_to_conversations_split_parameter(
    mock_user_prompt,
    mock_system_prompt,
    sample_examples,
):
    """Test split parameter is accepted (currently unused but valid)."""
    with patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_SYSTEM_PROMPT",
        "System.",
    ), patch(
        "src.llm_finetune.data_loading_utils.conversation.FACT_CHECK_USER_PROMPT",
        "Q: {claim}\nE: {evidence}",
    ):
        result_train = convert_to_conversations(
            sample_examples,
            split="train",
        )
        result_test = convert_to_conversations(
            sample_examples,
            split="test",
        )

    assert len(result_train) == len(result_test)
