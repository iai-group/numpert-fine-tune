"""conversation.py.

Convert expanded perturbation examples into the conversation format expected by
HuggingFace SFTTrainer.
"""

from __future__ import annotations

from typing import Dict, List

from src.numpert_prompt.prompt import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_PROMPT,
)


def convert_to_conversations(
    examples: List[Dict],
    split: str = "train",
    include_doc: bool = True,
) -> List[Dict]:
    """Convert examples to conversation format for SFTTrainer.

    Args:
        examples: List of expanded examples (from :func:`expand_perturbations`).
        split: ``"train"`` or ``"validation"`` / ``"test"``.
        include_doc: Whether to include the evidence document in the prompt.

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
        user_content = FACT_CHECK_USER_PROMPT.format(
            examples="",
            claim=example["claim"],
            evidence=evidence,
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
        # Legacy field, keep for backward compat if present
        if "perturbation" in example:
            entry["perturbation"] = example["perturbation"]
        messages.append(entry)

    return messages
