from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)

AFRICA_SPACED = (
    'Claim "Almost one in 10 teenage deaths are suicide." Verdict incorrect '
    "Africa Check asked the Psychiatry Management Group for the research."
)
AFRICA_GLUED = (
    '"80% of the population occupies less than 10% of land." Verdict '
    "incorrectThe EFF bases its claim on ownership data."
)
# Corpus form: the "Verdict: <rating>." label is its own sentence (measured
# max 4 chars beyond the label itself), so sentence mode removes exactly it.
CHECKYOURFACT = (
    "A post shows citizens blocking a NATO column. Verdict: Misleading. "
    "While the video is from Moldova, it is from 2016."
)


def test_token_spaced_africacheck():
    res = strip(AFRICA_SPACED, "africacheck.org")
    assert "Verdict incorrect" not in res.cleaned
    assert "Africa Check asked the Psychiatry" in res.cleaned
    assert res.removals[0].mode == "token"


def test_token_glued_repair_africacheck():
    res = strip(AFRICA_GLUED, "africacheck.org")
    assert "Verdict" not in res.cleaned
    assert "incorrectThe" not in res.cleaned
    assert "The EFF bases its claim" in res.cleaned


def test_sentence_checkyourfact_line():
    res = strip(CHECKYOURFACT, "checkyourfact.com")
    assert "Verdict:" not in res.cleaned
    assert "Misleading" not in res.cleaned
    assert "While the video is from Moldova" in res.cleaned
    assert res.removals[0].mode == "sentence"
