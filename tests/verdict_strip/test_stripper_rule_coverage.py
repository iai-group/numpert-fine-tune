"""Coverage tests for rule broadenings surfaced by the LLM-locate spike.

Each case pins a verdict phrasing the original anchors missed: the verdict must
be removed AND the neighbouring evidence sentence preserved. Snippets mirror the
real no-rule-located samples (job 462822).
"""

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)


def test_politifact_we_rate_it():
    doc = "The data backs the governor. We rate it False."
    res = strip(doc, "politifact.com")
    assert "We rate it False" not in res.cleaned
    assert "data backs the governor" in res.cleaned


def test_politifact_we_rate_named_statement():
    doc = "Experts disputed the figures. We rate Perry's statement True."
    res = strip(doc, "politifact.com")
    assert "We rate Perry's statement True" not in res.cleaned
    assert "Experts disputed the figures" in res.cleaned


def test_politifact_we_award():
    doc = "He cited an outdated law. We award Huckabee a Barely True."
    res = strip(doc, "politifact.com")
    assert "We award Huckabee" not in res.cleaned
    assert "outdated law" in res.cleaned


def test_politifact_claim_is():
    doc = "There is no evidence of an eruption. The claim is False."
    res = strip(doc, "politifact.com")
    assert "The claim is False" not in res.cleaned
    assert "no evidence of an eruption" in res.cleaned


def test_snopes_rate_with_adverb():
    doc = (
        'The video was edited before the war. '
        'We thus rate this claim as "True."'
    )
    res = strip(doc, "snopes.com")
    assert "rate this claim" not in res.cleaned.lower()
    assert "edited before the war" in res.cleaned


def test_snopes_we_rank():
    doc = (
        "Both outlets exaggerated the data. "
        "We rank the claims made by both as false."
    )
    res = strip(doc, "snopes.com")
    assert "We rank the claims" not in res.cleaned
    assert "exaggerated the data" in res.cleaned


def test_africacheck_rate_this_claim_sentence():
    doc = (
        "Poverty numbers are debated by economists. "
        "We therefore rate this claim as misleading."
    )
    res = strip(doc, "africacheck.org")
    assert "rate this claim" not in res.cleaned.lower()
    assert "debated by economists" in res.cleaned


def test_thequint_claim_is():
    doc = "The image is from 2018. Is it true?: The claim is false."
    res = strip(doc, "thequint.com")
    assert "The claim is false" not in res.cleaned
    assert "image is from 2018" in res.cleaned


def test_politifact_plain_evidence_unchanged():
    # the broadened We-rate anchor must not fire without a rating word
    doc = "We rate limiters on servers to balance load across regions."
    res = strip(doc, "politifact.com")
    assert res.cleaned == doc


# --- colon-form "Verdict:" leakage pass (2026-07-03) -----------------------


def test_africacheck_colon_verdict_label_and_restatement_removed():
    # Real shape (train 30, multi-claim report): "Verdict: Correct <verdict
    # restatement sentence>" — both leak; following evidence survives.
    doc = (
        "Statistics South Africa puts the expanded unemployment rate at "
        "36.8% in 2018. Verdict: Correct The EFF's claim that more than 9 "
        "million South Africans who need jobs are unemployed is correct. "
        "The next claim concerns school dropout rates."
    )
    res = strip(doc, "africacheck.org")
    assert "36.8% in 2018." in res.cleaned
    assert "Verdict: Correct" not in res.cleaned
    assert "is correct" not in res.cleaned
    assert "school dropout rates" in res.cleaned


def test_ferret_verdict_header_removed():
    # Real shape (train 1216): "Ferret Fact Service verdict: True <verdict
    # explanation sentence>".
    doc = (
        "Rail fares will rise by 3.1 percent under the retail price index. "
        "Ferret Fact Service verdict: True Sarwar is right to say that "
        "both rail fares and water charges are increasing. Water charges "
        "were confirmed in the budget."
    )
    res = strip(doc, "theferret.scot")
    assert "3.1 percent" in res.cleaned
    assert "verdict: True" not in res.cleaned
    assert "Sarwar is right to say" not in res.cleaned
    assert "confirmed in the budget." in res.cleaned


def test_ghanafact_colon_verdict_token_removed():
    # Real shape (train 6523): "Source: Facebook post Verdict: FALSE
    # Researched Gifty Tracy Aminu ..." — glued text, so token mode only
    # removes the label itself.
    doc = (
        "Five COVID-19 vaccines were previously announced. Source: "
        "Facebook post Verdict: FALSE Researched Gifty Tracy Aminu A "
        "widely shared Facebook post is claiming the effectiveness "
        "changed."
    )
    res = strip(doc, "ghanafact.com")
    assert "Verdict: FALSE" not in res.cleaned
    assert "widely shared Facebook post" in res.cleaned


def test_dw_bare_verdict_label_removed():
    # Real shape (train 7214): a bare "Verdict: False." line between the
    # claim and the explanation.
    doc = (
        "An X-ray image of the truck is said to show its explosive "
        "cargo.\n\nVerdict: False.\n\nThe truck in the X-ray photo isn't "
        "identical with the truck in the video."
    )
    res = strip(doc, "dw.com")
    assert "Verdict: False." not in res.cleaned
    assert "explosive cargo." in res.cleaned
    assert "isn't identical" in res.cleaned


def test_our_verdict_colon_removed_politifact_snopes():
    # Real shapes: trailing "Our verdict: Mostly True." (politifact) and
    # "Our verdict: Unproven." (snopes).
    doc_p = ("We believe that additional information is necessary to "
             "examine this claim. Our verdict: Mostly True.")
    res = strip(doc_p, "politifact.com")
    assert "Our verdict" not in res.cleaned
    assert "additional information is necessary" in res.cleaned

    doc_s = ("We have a plethora of head shots purporting to show a family "
             "resemblance between Clinton and Williams. Our verdict: "
             "Unproven.")
    res = strip(doc_s, "snopes.com")
    assert "Our verdict" not in res.cleaned
    assert "head shots" in res.cleaned
