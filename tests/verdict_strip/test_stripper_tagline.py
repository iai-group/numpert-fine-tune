from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)

JHOOTH = (
    "The Times also confirmed that the daily did not write or publish such a "
    "story. JHOOTH BOLE KAUVA KAATE The number of crows determines the "
    "intensity of the lie."
)


def test_tagline_removed_evidence_kept():
    res = strip(JHOOTH, "indiatoday.in")
    assert "JHOOTH" not in res.cleaned
    assert "number of crows" not in res.cleaned
    assert res.cleaned.endswith("publish such a story.")
    assert len(res.removals) == 1
    assert res.removals[0].rule == "jhooth_tagline"
    assert res.removals[0].mode == "tagline"


def test_no_tagline_no_change():
    text = "A plain evidence sentence with no verdict boilerplate."
    res = strip(text, "indiatoday.in")
    assert res.cleaned == text
    assert res.removals == []
