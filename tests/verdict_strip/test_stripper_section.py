from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)

POLITIFACT = (
    "Pence implied that Clinton dawdled before sending help. "
    "Our ruling Pence claimed that Clinton took 13 hours to send help. "
    "We rate this claim False."
)
AAP = (
    "Experts told AAP the claim was wrong. The Verdict The claim that Mike "
    "Pence acted improperly is false. US constitutional experts say so."
)


def test_section_cut_to_end_politifact():
    res = strip(POLITIFACT, "politifact.com")
    assert res.cleaned == (
        "Pence implied that Clinton dawdled before sending help."
    )
    assert "Our ruling" not in res.cleaned
    assert "We rate" not in res.cleaned
    assert any(r.mode == "section" for r in res.removals)


def test_section_cut_to_end_aap():
    res = strip(AAP, "aap.com.au")
    assert res.cleaned == "Experts told AAP the claim was wrong."
    assert "Verdict" not in res.cleaned


def test_no_anchor_no_change():
    text = "Just regular evidence about a ruling body's decision in court."
    # 'Our ruling' anchor must not fire on the word 'ruling' alone
    res = strip(text, "politifact.com")
    assert res.cleaned == text
