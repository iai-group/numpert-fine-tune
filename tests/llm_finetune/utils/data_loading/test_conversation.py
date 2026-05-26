"""Tests for conversation.py module."""

from __future__ import annotations

import pytest

from src.llm_finetune.utils.data_loading.conversation import (
    convert_to_conversations,
)
from src.prompt.prompt import PromptType


@pytest.fixture
def sample_examples():
    """Two examples covering True and False labels."""
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


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestConvertToConversationsBasic:
    """Test basic output structure and field correctness."""

    def test_returns_correct_count(self, sample_examples):
        result = convert_to_conversations(sample_examples)
        assert len(result) == 2

    def test_required_keys_present(self, sample_examples):
        result = convert_to_conversations(sample_examples)
        for entry in result:
            assert {
                "conversations",
                "claim_id",
                "perturbation_type",
                "label",
            } <= entry.keys()

    def test_conversation_has_three_turns(self, sample_examples):
        result = convert_to_conversations([sample_examples[0]])
        assert len(result[0]["conversations"]) == 3

    def test_conversation_roles(self, sample_examples):
        conv = convert_to_conversations([sample_examples[0]])[0][
            "conversations"
        ]
        assert conv[0]["role"] == "system"
        assert conv[1]["role"] == "user"
        assert conv[2]["role"] == "assistant"

    def test_label_true_converted(self, sample_examples):
        result = convert_to_conversations([sample_examples[0]])
        assert result[0]["label"] == '{"label": true}'
        assert result[0]["conversations"][2]["content"] == '{"label": true}'

    def test_label_false_converted(self, sample_examples):
        result = convert_to_conversations([sample_examples[1]])
        assert result[0]["label"] == '{"label": false}'
        assert result[0]["conversations"][2]["content"] == '{"label": false}'

    def test_include_doc_true_embeds_evidence(self, sample_examples):
        result = convert_to_conversations(
            [sample_examples[0]], include_doc=True
        )
        user_content = result[0]["conversations"][1]["content"]
        assert sample_examples[0]["doc"] in user_content

    def test_include_doc_false_omits_evidence(self, sample_examples):
        result = convert_to_conversations(
            [sample_examples[0]], include_doc=False
        )
        user_content = result[0]["conversations"][1]["content"]
        assert sample_examples[0]["doc"] not in user_content

    def test_preserves_claim_id(self, sample_examples):
        result = convert_to_conversations(sample_examples)
        for i, entry in enumerate(result):
            assert entry["claim_id"] == sample_examples[i]["claim_id"]

    def test_preserves_perturbation_type(self, sample_examples):
        result = convert_to_conversations(sample_examples)
        for i, entry in enumerate(result):
            assert (
                entry["perturbation_type"]
                == sample_examples[i]["perturbation_type"]
            )

    def test_empty_input_returns_empty_list(self):
        assert convert_to_conversations([]) == []

    def test_split_parameter_accepted(self, sample_examples):
        train = convert_to_conversations(sample_examples, split="train")
        test = convert_to_conversations(sample_examples, split="test")
        assert len(train) == len(test)

    def test_optional_source_file_passed_through(self):
        example = {
            "claim_id": "c1",
            "claim": "claim text",
            "doc": "doc text",
            "label": "True",
            "perturbation_type": "ORIGINAL",
            "source_file": "data/test.json",
        }
        result = convert_to_conversations([example])
        assert result[0]["source_file"] == "data/test.json"

    def test_optional_perturbation_method_passed_through(self):
        example = {
            "claim_id": "c1",
            "claim": "claim text",
            "doc": "doc text",
            "label": "True",
            "perturbation_type": "ORIGINAL",
            "perturbation_method": "numeration",
        }
        result = convert_to_conversations([example])
        assert result[0]["perturbation_method"] == "numeration"


# ---------------------------------------------------------------------------
# Prompt-type tests
# ---------------------------------------------------------------------------


class TestConvertToConversationsPromptType:
    """Test the prompt_type parameter controls which examples are injected."""

    def test_default_prompt_type_is_zero_shot(self, sample_examples):
        """Calling without prompt_type should behave identically to
        ZERO_SHOT."""
        default = convert_to_conversations([sample_examples[0]])
        explicit = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.ZERO_SHOT
        )
        assert (
            default[0]["conversations"][1]["content"]
            == explicit[0]["conversations"][1]["content"]
        )

    def test_zero_shot_contains_no_few_shot_examples(self, sample_examples):
        """ZERO_SHOT injects no example text."""
        result = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.ZERO_SHOT
        )
        user_content = result[0]["conversations"][1]["content"]
        # FEW_SHOT_EXAMPLES has the oil-spill claim — must not appear
        assert "150,000 (oil spill) claims" not in user_content

    def test_few_shot_contains_oil_spill_example(self, sample_examples):
        """FEW_SHOT injects FEW_SHOT_EXAMPLES which includes the oil-spill
        claim."""
        result = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.FEW_SHOT
        )
        user_content = result[0]["conversations"][1]["content"]
        assert "150,000 (oil spill) claims" in user_content

    def test_few_shot_contains_dicaprio_example(self, sample_examples):
        """FEW_SHOT injects FEW_SHOT_EXAMPLES which includes the DiCaprio
        claim."""
        result = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.FEW_SHOT
        )
        user_content = result[0]["conversations"][1]["content"]
        assert "Leonardo DiCaprio" in user_content

    def test_negative_synthetic_with_reasoning_contains_eiffel_tower(
        self, sample_examples
    ):
        """NEGATIVE_SYNTHETIC_ALL_WITH_REASONING injects the Eiffel Tower
        example."""
        result = convert_to_conversations(
            [sample_examples[0]],
            prompt_type=PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING,
        )
        user_content = result[0]["conversations"][1]["content"]
        assert "Eiffel Tower" in user_content

    def test_negative_synthetic_with_reasoning_contains_reasoning_chain(
        self, sample_examples
    ):
        """Verify NEGATIVE_SYNTHETIC_ALL_WITH_REASONING injects reasoning."""
        result = convert_to_conversations(
            [sample_examples[0]],
            prompt_type=PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING,
        )
        user_content = result[0]["conversations"][1]["content"]
        assert "Reasoning:" in user_content  # noqa: E501

    def test_zero_shot_and_few_shot_differ(self, sample_examples):
        """ZERO_SHOT and FEW_SHOT produce different user-message content."""
        zero = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.ZERO_SHOT
        )[0]["conversations"][1]["content"]
        few = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.FEW_SHOT
        )[0]["conversations"][1]["content"]
        assert zero != few

    def test_few_shot_and_negative_synthetic_differ(self, sample_examples):
        """FEW_SHOT and NEGATIVE_SYNTHETIC_ALL_WITH_REASONING are distinct."""
        few = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.FEW_SHOT
        )[0]["conversations"][1]["content"]
        neg = convert_to_conversations(
            [sample_examples[0]],
            prompt_type=PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING,
        )[0]["conversations"][1]["content"]
        assert few != neg

    def test_prompt_type_applied_to_all_examples(self, sample_examples):
        """When a batch is converted, every entry gets the same examples
        block."""
        result = convert_to_conversations(
            sample_examples, prompt_type=PromptType.FEW_SHOT
        )
        for entry in result:
            assert (
                "150,000 (oil spill) claims"
                in entry["conversations"][1]["content"]
            )

    def test_few_shot_examples_injected_verbatim(self, sample_examples):
        """The full FEW_SHOT_EXAMPLES string is present in the formatted
        prompt."""
        result = convert_to_conversations(
            [sample_examples[0]], prompt_type=PromptType.FEW_SHOT
        )
        user_content = result[0]["conversations"][1]["content"]
        # A distinctive multi-word phrase from FEW_SHOT_EXAMPLES
        assert "Gulf Coast Claims Facility" in user_content

    def test_negative_synthetic_with_reasoning_verbatim(self, sample_examples):
        """The full NEGATIVE_EXAMPLES_SYNTHETIC_WITH_REASONING content is
        present."""
        result = convert_to_conversations(
            [sample_examples[0]],
            prompt_type=PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING,
        )
        user_content = result[0]["conversations"][1]["content"]
        assert "three hundred and fifty-one meters tall" in user_content

    def test_claim_and_evidence_still_present_with_few_shot(
        self, sample_examples
    ):
        """The actual claim and evidence appear alongside the injected
        examples."""
        result = convert_to_conversations(
            [sample_examples[0]],
            include_doc=True,
            prompt_type=PromptType.FEW_SHOT,
        )
        user_content = result[0]["conversations"][1]["content"]
        assert sample_examples[0]["claim"] in user_content
        assert sample_examples[0]["doc"] in user_content

    def test_prompt_type_does_not_affect_assistant_turn(self, sample_examples):
        """prompt_type changes only the user turn; assistant label is
        unaffected."""
        for pt in (
            PromptType.ZERO_SHOT,
            PromptType.FEW_SHOT,
            PromptType.NEGATIVE_SYNTHETIC_ALL_WITH_REASONING,
        ):
            result = convert_to_conversations(
                [sample_examples[0]], prompt_type=pt
            )
            assert result[0]["conversations"][2]["content"] == '{"label": true}'
