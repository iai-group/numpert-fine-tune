"""verdict-strip v3 rules (2026-07-08 pass).

Fixtures are real corpus snippets from the PRISTINE QuanTemp corpus
(train/val/test_claims_quantemp.json). Each rule gets a positive test
(structural artifact removed, adjacent evidence kept) and a no-match
negative (prose / wrong domain survives untouched).
"""

import pytest

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)


# =====================================================================
# Task 1a -- KEY TAKE AWAY structural header (feedback domains, token)
# =====================================================================


@pytest.mark.parametrize(
    "domain",
    ["healthfeedback.org", "sciencefeedback.co", "climatefeedback.org"],
)
def test_key_take_away_header_removed(domain):
    # healthfeedback id=8 form.
    doc = (
        "This sequence can be found in other living things, showing that "
        "it occurs naturally. KEY TAKE AWAY The claim is based on the "
        "observation that the spike protein of SARS-CoV-2 shares a short "
        "sequence with a patented gene."
    )
    res = strip(doc, domain)
    assert "KEY TAKE AWAY" not in res.cleaned
    assert "it occurs naturally." in res.cleaned
    assert "The claim is based on the observation" in res.cleaned
    assert any(
        r.rule == "key_take_away" and r.mode == "token"
        for r in res.removals
    )


def test_key_take_away_lowercase_prose_survives():
    # Case-sensitive: prose "key takeaway" (the politifact/snopes/fullfact
    # stragglers) must survive on a feedback domain too.
    doc = (
        "The key takeaway from our exercise is not the specific pathogen "
        "but the need for preparedness."
    )
    res = strip(doc, "healthfeedback.org")
    assert res.cleaned == doc
    assert res.removals == []


# =====================================================================
# Task 1b -- boomlive "A FactCheck" + bare "FACT-CHECK" header (token)
# =====================================================================


def test_boomlive_a_factcheck_and_fact_check_header_removed():
    # boomlive id=190 form: headline suffix glued to the FACT-CHECK header.
    doc = (
        "Was Akhilesh Yadav Warned To Join SP If Not Made UP CM Again? "
        "A FactCheck FACT-CHECK BOOM found that the viral quote being "
        "attributed to Akhilesh Yadav is fabricated."
    )
    res = strip(doc, "boomlive.in")
    assert "A FactCheck" not in res.cleaned
    assert "FACT-CHECK" not in res.cleaned
    assert "BOOM found that the viral quote" in res.cleaned
    assert any(
        r.rule == "a_factcheck_header" and r.mode == "token"
        for r in res.removals
    )


def test_boomlive_glued_fact_check_header_removed():
    # Glued form "archive.FACT-CHECKBOOM" -> "archive. BOOM ...".
    doc = (
        "Click here for an archive.FACT-CHECKBOOM analysed the video "
        "and found it was shot in 2018."
    )
    res = strip(doc, "boomlive.in")
    assert "FACT-CHECK" not in res.cleaned
    assert "BOOM analysed the video" in res.cleaned


def test_boomlive_lowercase_fact_check_prose_survives():
    doc = (
        "The organisation ran a fact check on the viral claim and "
        "published a detailed report."
    )
    res = strip(doc, "boomlive.in")
    assert res.cleaned == doc
    assert res.removals == []


# =====================================================================
# Task 1c -- climatefeedback "REVIEW CLAIM:" header (token, header only)
# =====================================================================


def test_review_claim_header_removed_claim_restatement_kept():
    # climatefeedback id=263 form. token: header goes, the claim
    # restatement (which duplicates the record's claim field) stays.
    doc = (
        "the scientists behind the studies have confirmed. REVIEW CLAIM: "
        "'Global warming' is a myth - so say 80 graphs from 58 "
        "peer-reviewed scientific papers published in 2017. Patrick "
        "Brown, Assistant Professor: These studies do not challenge the "
        "consensus."
    )
    res = strip(doc, "climatefeedback.org")
    assert "REVIEW CLAIM:" not in res.cleaned
    # token mode keeps the claim restatement AND the scientist rebuttal
    # (label_next would over-remove into "Patrick Brown ..." here).
    assert "'Global warming' is a myth" in res.cleaned
    assert "Patrick Brown, Assistant Professor" in res.cleaned
    assert any(
        r.rule == "review_claim" and r.mode == "token"
        for r in res.removals
    )


def test_review_claim_no_match_unchanged():
    doc = "The article reviews the claim about global temperatures rising."
    res = strip(doc, "climatefeedback.org")
    assert res.cleaned == doc
    assert res.removals == []


# =====================================================================
# Task 1d -- "Fact Check:" structural section header (six domains, token)
# =====================================================================


FACT_CHECK_FIXTURES = [
    (
        "checkyourfact.com",
        "The video is from 2016.\n\nFact Check:\n\nUkraine applied for NATO "
        "membership last week after Russia announced the annexation.",
        "Ukraine applied for NATO membership",
    ),
    (
        # In production claim_review_block (footer, runs first) consumes
        # this trailing block; this fixture exercises the token-rule
        # fallback (mid-doc headers, e.g. train:818).
        "vishvasnews.com",
        "Two jets were shot down and two pilots were killed. Claimed By : "
        "FB User Adnan Khan Fact Check : False",
        "were killed.",
    ),
    (
        "indiatoday.in",
        "Fact Check: 50% seats in NIT-Goa reserved for home students, is "
        "it an exception?",
        "50% seats in NIT-Goa",
    ),
    (
        "newsmeter.in",
        "Also Read Cream of Tartar does not provide relief from migraine "
        "Fact Check: The propaganda that the Central Government is closing "
        "schools is false.",
        "The propaganda that the Central Government",
    ),
    (
        "snopes.com",
        'Chayan Kundu et al. "Fact Check: You Can Sleep through This Viral '
        'Claim." India Today.',
        "You Can Sleep through This Viral Claim",
    ),
    (
        "boomlive.in",
        "The image is a fabrication. Also Read: Fact Check: Did Elon Musk "
        "Release A 'Tesla Smartphone'?",
        "Did Elon Musk Release",
    ),
]


@pytest.mark.parametrize(
    "domain,doc,kept",
    FACT_CHECK_FIXTURES,
    ids=[f[0] for f in FACT_CHECK_FIXTURES],
)
def test_fact_check_header_removed(domain, doc, kept):
    res = strip(doc, domain)
    assert "Fact Check:" not in res.cleaned
    assert "Fact Check :" not in res.cleaned
    assert kept in res.cleaned
    assert any(
        r.rule == "fact_check_header" and r.mode == "token"
        for r in res.removals
    )


def test_fact_check_lowercase_prose_survives():
    doc = "The reporter did a fact check: it turned out to be accurate."
    res = strip(doc, "checkyourfact.com")
    assert res.cleaned == doc
    assert not any(
        r.rule == "fact_check_header" for r in res.removals
    )


def test_fact_check_wrong_domain_survives():
    # dw.com has "Fact Check:" too, but is NOT one of the six target
    # domains, so the header rule must not fire there.
    doc = (
        "DW Fact Check: The photo is genuine. It was taken in Kyiv in "
        "2022."
    )
    res = strip(doc, "dw.com")
    assert not any(
        r.rule == "fact_check_header" for r in res.removals
    )


# =====================================================================
# Task 2 -- "VERDICT SOURCE: ... <year>" metadata line (feedback, token)
# =====================================================================


@pytest.mark.parametrize(
    "domain",
    ["healthfeedback.org", "sciencefeedback.co", "climatefeedback.org"],
)
def test_verdict_source_line_removed_through_year(domain):
    # Isolated (no DETAILS tail) so the verdict_source removal is exact:
    # from "VERDICT SOURCE" through the first 4-digit year.
    doc = (
        "The sequence occurs naturally. VERDICT SOURCE: Connor Boyd, "
        "Daily Mail, 23 Feb. 2022 The remaining evidence text stands "
        "here unaffected."
    )
    res = strip(doc, domain)
    assert "VERDICT SOURCE" not in res.cleaned
    assert "Connor Boyd" not in res.cleaned
    assert "2022" not in res.cleaned
    assert "The sequence occurs naturally." in res.cleaned
    assert "The remaining evidence text stands here unaffected." in (
        res.cleaned
    )
    assert any(
        r.rule == "verdict_source" and r.mode == "token"
        for r in res.removals
    )


def test_verdict_source_multiple_outlets_stops_at_first_year():
    # id=347 form: a long multi-outlet source list, lazy match ends at the
    # publication year, not any earlier digits. The trailing "DETAILS
    # Overstates scientific confidence: <explanation>" is removed by
    # details_rating (label + explanation sentence); the evidence sentence
    # after it survives.
    doc = (
        "The study contains flaws. VERDICT SOURCE: Daniel Horowitz, Jim "
        "Hoft, Gateway Pundit, The Blaze, Zero Hedge, 12 Dec. 2021 "
        "DETAILS Overstates scientific confidence: The study did not "
        "support the claim. Independent trials confirmed the vaccine's "
        "safety."
    )
    res = strip(doc, "healthfeedback.org")
    assert "VERDICT SOURCE" not in res.cleaned
    assert "Zero Hedge" not in res.cleaned
    assert "12 Dec. 2021" not in res.cleaned
    assert "DETAILS" not in res.cleaned
    assert "Overstates scientific confidence" not in res.cleaned
    assert "The study did not support the claim." not in res.cleaned
    assert "Independent trials confirmed the vaccine's safety." in (
        res.cleaned
    )
    assert any(r.rule == "verdict_source" for r in res.removals)
    assert any(r.rule == "details_rating" for r in res.removals)


def test_verdict_source_no_year_left_unmatched():
    # No 4-digit year within reach -> rule must not fire (leave untouched).
    doc = (
        "The claim is unsupported. VERDICT SOURCE: Anonymous social media "
        "user posting without any date attached to the message."
    )
    res = strip(doc, "healthfeedback.org")
    assert not any(r.rule == "verdict_source" for r in res.removals)
    assert "VERDICT SOURCE: Anonymous" in res.cleaned


# =====================================================================
# Task 3 -- factcheckhub author-bio block -> end-of-doc (footer, section)
# =====================================================================


def test_factcheckhub_author_bio_block_removed_to_eod():
    # id=1235 form: the bio runs from the author NAME to end-of-doc.
    doc = (
        "The footage has been online since January prior to the primary "
        "election.\n\nNurudeen Akewushola is a fact-checker with "
        "FactCheckHub. He has authored several fact checks which have "
        "contributed to the fight against information disorder. You can "
        "reach him via nyahaya@icirnigeria.org."
    )
    res = strip(doc, "factcheckhub.com")
    # The sentence-start pull-back pulls the author name into the cut.
    assert "Nurudeen Akewushola" not in res.cleaned
    assert "fact-checker with FactCheckHub" not in res.cleaned
    assert "You can reach him via" not in res.cleaned
    assert (
        "online since January prior to the primary election."
        in res.cleaned
    )
    assert any(
        r.rule == "factcheckhub_bio_block"
        and r.mode == "section"
        and r.category == "footer"
        for r in res.removals
    )


def test_factcheckhub_bio_block_the_variant_removed():
    # "with The FactCheckHub" variant, hyphenless "factchecker".
    doc = (
        "The viral message is misleading. Ada Obi is a factchecker with "
        "The FactCheckHub. She authored this report."
    )
    res = strip(doc, "factcheckhub.com")
    assert "Ada Obi" not in res.cleaned
    assert "The viral message is misleading." in res.cleaned
    assert any(
        r.rule == "factcheckhub_bio_block" for r in res.removals
    )


def test_factcheckhub_bio_block_negative_no_mention():
    doc = (
        "The bridge was inaugurated in 2021 according to state media. "
        "The report cited official records."
    )
    res = strip(doc, "factcheckhub.com")
    assert res.cleaned == doc
    assert res.removals == []


# =====================================================================
# Task 4 -- "Claim Review:" trailing block -> end-of-doc (footer, section)
# =====================================================================


def test_vishvasnews_claim_review_block_removed_to_eod():
    # id=127 form: "Claim Review :" opens the trailing metadata block
    # (claim restatement + Claimed By + Fact Check verdict + signoff).
    doc = (
        "The images are from old, unrelated events and the claim is "
        "fake. Claim Review : Indian Air force (IAF) had conducted an "
        "airstrike in India-Nepal border. Claimed By : FB User Adnan "
        "Khan Fact Check : False Know the truth! Help us in our fight "
        "against fake news. Tags"
    )
    res = strip(doc, "vishvasnews.com")
    assert "Claim Review" not in res.cleaned
    assert "Claimed By" not in res.cleaned
    assert "Fact Check : False" not in res.cleaned
    assert "Know the truth!" not in res.cleaned
    assert (
        "The images are from old, unrelated events and the claim is fake."
        in res.cleaned
    )
    assert any(
        r.rule == "claim_review_block"
        and r.mode == "section"
        and r.category == "footer"
        for r in res.removals
    )


def test_newschecker_claim_review_block_removed_to_eod():
    # id=6344 form: same trailing-block shape (no space before colon).
    doc = (
        "The clip is from 2017 when Pakistan won the Championship Trophy "
        "and not 2021. Result: Misleading Claim Review: Celebrations in "
        "Srinagar post Ind Vs Pak T20 WC match. Claimed By: Social Media "
        "Users Fact Check: Misleading Our Sources Social media If you "
        "would like us to fact-check a claim, email us."
    )
    res = strip(doc, "newschecker.in")
    assert "Claim Review" not in res.cleaned
    assert "Claimed By" not in res.cleaned
    assert "If you would like us to fact-check" not in res.cleaned
    assert (
        "The clip is from 2017 when Pakistan won the Championship Trophy "
        "and not 2021." in res.cleaned
    )
    assert any(
        r.rule == "claim_review_block"
        and r.mode == "section"
        and r.category == "footer"
        for r in res.removals
    )


def test_claim_review_negative_no_block():
    doc = (
        "The reviewer examined the claim and found supporting documents "
        "in the public archive."
    )
    res = strip(doc, "vishvasnews.com")
    assert res.cleaned == doc
    assert not any(
        r.rule == "claim_review_block" for r in res.removals
    )


# =====================================================================
# Review follow-up -- _FEEDBACK_LABEL missing rating forms (label_next)
# =====================================================================
# Six DETAILS-style rating labels leaked on the feedback trio. Fixtures
# are real corpus label + explanation snippets; the evidence sentence
# after the explanation must survive.


FEEDBACK_LABEL_FIXTURES = [
    (
        # healthfeedback id=2329 (singular; plural already covered).
        "misrepresents_source_singular",
        "healthfeedback.org",
        "DETAILS Misrepresents source: The study by researchers in "
        "Denmark didn't conclude that vaccination increases infection "
        "risk. The data actually show the opposite trend.",
        "Misrepresents source",
        "The data actually show the opposite trend.",
    ),
    (
        # sciencefeedback id=1197.
        "fails_to_grasp",
        "sciencefeedback.co",
        "DETAILS Fails to grasp significance of observation: The data "
        "provided by the Israeli Ministry of Health actually points to "
        "high vaccine effectiveness. Later surveys confirmed this.",
        "Fails to grasp significance of observation",
        "Later surveys confirmed this.",
    ),
    (
        # healthfeedback id=347.
        "overstates_confidence",
        "healthfeedback.org",
        "DETAILS Overstates scientific confidence: The study contained "
        "multiple methodological flaws that call the reliability into "
        "question. Independent replication attempts failed.",
        "Overstates scientific confidence",
        "Independent replication attempts failed.",
    ),
    (
        # healthfeedback id=2323 ("lacks context" variant already covered).
        "lack_of_context",
        "healthfeedback.org",
        "DETAILS Lack of context: It is unclear whether ejaculation "
        "reduces the risk of prostate cancer. Studies report conflicting "
        "associations.",
        "Lack of context",
        "Studies report conflicting associations.",
    ),
    (
        # healthfeedback id=2089; appears without the DETAILS prefix too.
        "misrepresents_complex_reality",
        "healthfeedback.org",
        "COVID-19 vaccines don't alter people's DNA. Misrepresents a "
        "complex reality: The age-standardized rates of COVID-19 "
        "hospitalizations are higher among the unvaccinated. Raw counts "
        "alone are not informative.",
        "Misrepresents a complex reality",
        "Raw counts alone are not informative.",
    ),
    (
        # healthfeedback id=208 (no DETAILS prefix, capital hyphenated).
        "cherry_picking",
        "healthfeedback.org",
        "The drug saw widespread usage among the population. "
        "Cherry-picking: The authors conveniently excluded from the "
        "hydroxychloroquine group certain patients. The exclusions "
        "changed the mortality figures.",
        "Cherry-picking",
        "The exclusions changed the mortality figures.",
    ),
]


@pytest.mark.parametrize(
    "name,domain,doc,label_gone,evidence_kept",
    [(f[0], f[1], f[2], f[3], f[4]) for f in FEEDBACK_LABEL_FIXTURES],
    ids=[f[0] for f in FEEDBACK_LABEL_FIXTURES],
)
def test_feedback_label_new_forms_removed(
    name, domain, doc, label_gone, evidence_kept
):
    res = strip(doc, domain)
    assert label_gone not in res.cleaned
    assert "DETAILS" not in res.cleaned
    assert evidence_kept in res.cleaned
    assert any(
        r.rule == "details_rating" and r.mode == "label_next"
        for r in res.removals
    )


def test_cherry_picking_lowercase_prose_survives():
    # climatefeedback id=6510: lowercase "cherry picking:" is PROSE (a
    # quoted explanation), not a rating label. The label alternative is
    # case-sensitive hyphenated "Cherry-picking", so this must survive.
    doc = (
        "Morrison's reliance upon this time period is an example of "
        "cherry picking: selectively picking a particularly warm year "
        "as a baseline in order to downplay the warming trend."
    )
    res = strip(doc, "climatefeedback.org")
    assert "cherry picking: selectively picking" in res.cleaned
    assert not any(r.rule == "details_rating" for r in res.removals)
