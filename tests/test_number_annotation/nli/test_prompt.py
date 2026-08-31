"""Tests for NLI prompt construction."""

from src.number_annotation.core.schema import NumberSpan
from src.annotation_prompt.prompt import (
    NLI_SYSTEM_PROMPT,
    build_nli_user_prompt,
)


def _claim_num(id, surface):
    return NumberSpan(
        id=id, surface=surface, char_start=0, char_end=0, value=None,
        value_low=None, value_high=None, value_type="count", unit=None,
        modifier="exact", sign="+", context="",
    )


def test_system_prompt_mentions_scale_and_relations():
    assert "1" in NLI_SYSTEM_PROMPT and "5" in NLI_SYSTEM_PROMPT
    for rel in ("supports", "refutes", "neutral"):
        assert rel in NLI_SYSTEM_PROMPT


def test_user_prompt_contains_claim_numbers_and_sentence():
    claim = "More than 50 percent of immigrants use welfare."
    claim_numbers = [_claim_num(0, "More than 50 percent")]
    marked = "The study found [N0]about 40%[/N0] used welfare."
    out = build_nli_user_prompt(claim, claim_numbers, marked)
    assert claim in out
    assert "[N0]about 40%[/N0]" in out
    assert "More than 50 percent" in out  # claim-number listing
    assert "c0" in out  # claim numbers labelled c0, c1, ...


def test_user_prompt_handles_no_claim_numbers():
    out = build_nli_user_prompt("A claim.", [], "Sentence [N0]5[/N0].")
    assert "A claim." in out
    assert "[N0]5[/N0]" in out
