from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)

AFP = (
    "The photo was widely shared on Facebook. However, the claim is false. "
    "A reverse image search found the photo was published by AFP in 2017."
)
FACTLY = (
    "There is varying data about the toilets built. Hence the claim is "
    "MISLEADING. The second claim is about sanitation coverage."
)
# Regression for the wrong-sentence bug: the anchor's leading `[^.]*` stops
# only at `.`, so it spans the `!`/`?` boundary PySBD does split on. The hit
# must be attributed to the verdict sentence (match end), not the preceding
# evidence sentence (match start) — otherwise the evidence is cut and the
# verdict leaks.
SNOPES_BANG = (
    "The image spread widely on social media! "
    "We rate this claim False."
)
AFP_QUESTION = (
    "Was the viral video filmed in Paris? "
    "A reverse image search shows it is from Berlin. "
    "However, the claim is false."
)


def test_sentence_removed_neighbors_kept_afp():
    res = strip(AFP, "factcheck.afp.com")
    assert "the claim is false" not in res.cleaned.lower()
    assert "widely shared on Facebook" in res.cleaned
    assert "reverse image search" in res.cleaned
    assert any(r.mode == "sentence" for r in res.removals)


def test_sentence_removed_mid_doc_factly():
    res = strip(FACTLY, "factly.in")
    assert "claim is MISLEADING" not in res.cleaned
    assert "varying data about the toilets" in res.cleaned
    assert "second claim is about sanitation" in res.cleaned


def test_sentence_anchor_spanning_bang_keeps_evidence_removes_verdict():
    res = strip(SNOPES_BANG, "snopes.com")
    # verdict removed, evidence preserved (not the other way around)
    assert "we rate this claim" not in res.cleaned.lower()
    assert "image spread widely on social media" in res.cleaned


def test_sentence_idempotent_across_boundary():
    res1 = strip(SNOPES_BANG, "snopes.com")
    res2 = strip(res1.cleaned, "snopes.com")
    # no verdict left for a second pass to find
    assert res2.removals == []
    assert res2.cleaned == res1.cleaned


def test_sentence_anchor_spanning_question_keeps_evidence():
    res = strip(AFP_QUESTION, "factcheck.afp.com")
    assert "the claim is false" not in res.cleaned.lower()
    assert "filmed in Paris" in res.cleaned
    assert "reverse image search shows it is from Berlin" in res.cleaned


# Regression: politifact verdicts frequently end with "!" ("Pants on Fire!").
# The anchor must terminate on any sentence-ender ([.!?]) and not overshoot
# past it into the following footer sentence.
POLITIFACT_BANG = (
    "We rate this claim Pants on Fire! "
    "Send us a tip about a claim to fact-check."
)


def test_politifact_bang_verdict_removed_footer_kept():
    res = strip(POLITIFACT_BANG, "politifact.com")
    assert "We rate this claim" not in res.cleaned
    assert "Pants on Fire" not in res.cleaned
    assert "Send us a tip" in res.cleaned
    res2 = strip(res.cleaned, "politifact.com")
    assert res2.removals == []  # idempotent: no verdict left behind


def test_sentence_mode_bounds_removal_on_glued_missing_spaces():
    # Missing spaces after periods make PySBD merge sentences into one blob;
    # sentence-mode removal must cut ONLY the verdict sentence, not the glued
    # evidence before it. Regression for the id60 over-cut.
    doc = (
        "In 1970, the board's budget listed five staffers.But for an "
        "accurate comparison you have to throw in three committee clerks.So "
        "that's a total of eight back then.Today, the board has 38 staff "
        "positions.We rate Abele's statement Mostly True."
    )
    res = strip(doc, "politifact.com")
    assert "We rate Abele's statement Mostly True." not in res.cleaned
    assert "five staffers." in res.cleaned
    assert "three committee clerks." in res.cleaned
    assert "Today, the board has 38 staff positions." in res.cleaned
