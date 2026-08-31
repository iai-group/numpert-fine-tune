from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)

POLITIFACT_BOTH = (
    "Body of the fact check with evidence. "
    "Our ruling The claim restated. We rate this claim Mostly False."
)


def test_no_rule_domain_byte_identical():
    text = "Evidence with the word false used legitimately in a sentence."
    res = strip(text, "reuters.com")  # no rules registered
    assert res.cleaned == text
    assert res.removals == []


def test_idempotent():
    res1 = strip(POLITIFACT_BOTH, "politifact.com")
    res2 = strip(res1.cleaned, "politifact.com")
    assert res2.cleaned == res1.cleaned
    assert res2.removals == []


def test_multi_rule_section_runs_first():
    res = strip(POLITIFACT_BOTH, "politifact.com")
    assert res.cleaned == "Body of the fact check with evidence."
    # the section cut already removed the 'We rate' sentence -> only 1 removal
    # (reversed rule order would leave the sentence rule to also fire -> 2)
    assert len(res.removals) == 1
    assert res.removals[0].mode == "section"


def test_empty_doc_safe():
    assert strip("", "politifact.com").cleaned == ""
    assert strip("   ", "politifact.com").cleaned == "   "
