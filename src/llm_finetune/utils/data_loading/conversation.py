"""conversation.py.

Convert expanded perturbation examples into the conversation format expected by
HuggingFace SFTTrainer.
"""

from __future__ import annotations

from typing import Dict, List

from src.prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    PromptType,
    get_fact_check_user_prompt,
)


def convert_to_conversations(
    examples: List[Dict],
    split: str = "train",
    include_doc: bool = True,
    prompt_type: PromptType = PromptType.ZERO_SHOT,
) -> List[Dict]:
    """Convert examples to conversation format for SFTTrainer.

    Args:
        examples: List of expanded example dictionaries to convert.
        split: ``"train"`` or ``"validation"`` / ``"test"``.
        include_doc: Whether to include the evidence document in the prompt.
            When ``False``, the evidence field is passed as an empty string.
        prompt_type: Controls which example block is injected before the
            claim/evidence. Defaults to ``PromptType.ZERO_SHOT`` (no examples).

    Returns:
        List of examples in conversation format.
    """
    messages: List[Dict] = []

    for example in examples:
        conversations = []

        # System message
        conversations.append(
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT}
        )

        # User message
        evidence = example["doc"] if include_doc else ""
        user_content = get_fact_check_user_prompt(
            claim=example["claim"],
            evidence=evidence,
            prompt_type=prompt_type,
        )
        conversations.append({"role": "user", "content": user_content})

        # Convert label to JSON format
        json_label = (
            '{"label": true}'
            if example["label"] == "True"
            else '{"label": false}'
        )

        # Assistant response (always included — SFT needs it for training,
        # evaluation uses label for scoring)
        conversations.append({"role": "assistant", "content": json_label})

        entry: Dict = {
            "conversations": conversations,
            "claim_id": example["claim_id"],
            "perturbation_type": example["perturbation_type"],
            "label": json_label,
        }
        # Pass through optional metadata fields
        if "source_file" in example:
            entry["source_file"] = example["source_file"]
        if "perturbation_method" in example:
            entry["perturbation_method"] = example["perturbation_method"]
        if "sample_type" in example:
            entry["sample_type"] = example["sample_type"]
        # Legacy field, keep for backward compat if present
        if "perturbation" in example:
            entry["perturbation"] = example["perturbation"]
        messages.append(entry)

    return messages
