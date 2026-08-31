"""Residual-sweep verdict/footer rules (2026-07-03 pass)."""

import pytest

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    strip,
)


# --- apnews.com "ap_assessment" (token) ------------------------------------


def test_ap_assessment_glued_no_spaces():
    doc = (
        "Officers were filmed arresting Minneapolis police officers."
        "AP’S ASSESSMENT: False.Social media users are sharing a "
        "page from the autopsy report."
    )
    res = strip(doc, "apnews.com")
    assert "AP’S ASSESSMENT" not in res.cleaned
    assert "False." not in res.cleaned
    assert "arresting Minneapolis police officers." in res.cleaned
    assert "Social media users are sharing" in res.cleaned
    assert res.removals[0].mode == "token"
    assert res.removals[0].rule == "ap_assessment"


def test_ap_assessment_spaced_variant():
    doc = (
        "Officers made the arrest without incident. AP’S "
        "ASSESSMENT: Missing context. Social media posts claim the "
        "video shows a struggle."
    )
    res = strip(doc, "apnews.com")
    assert "AP’S ASSESSMENT" not in res.cleaned
    assert "Missing context." not in res.cleaned
    assert "Officers made the arrest without incident." in res.cleaned
    assert "Social media posts claim the video shows a struggle." in (
        res.cleaned
    )


# --- apnews.com / politifact.com "the_facts_header" (token, case-sens) ----


@pytest.mark.parametrize("domain", ["apnews.com", "politifact.com"])
def test_the_facts_header_removed(domain):
    doc = (
        "THE FACTS: A podcast clip shared on Instagram this week "
        "falsely claims the vaccine causes infertility."
    )
    res = strip(doc, domain)
    assert "THE FACTS" not in res.cleaned
    assert (
        "A podcast clip shared on Instagram this week falsely claims "
        "the vaccine causes infertility." in res.cleaned
    )
    assert any(r.rule == "the_facts_header" for r in res.removals)


@pytest.mark.parametrize("domain", ["apnews.com", "politifact.com"])
def test_the_facts_header_lowercase_prose_unchanged(domain):
    doc = (
        "According to the facts: the mayor stated the budget was "
        "balanced last year."
    )
    res = strip(doc, domain)
    assert res.cleaned == doc
    assert res.removals == []


# --- politifact.com "our_ruling" extension (section) -----------------------


def test_our_ruling_the_ruling_colon_removed():
    doc = (
        "Vaccines have undergone rigorous testing before approval. The "
        "ruling: The Oregon Department of Agriculture put out a news "
        "release stating pesticide levels were within safe limits."
    )
    res = strip(doc, "politifact.com")
    assert "The ruling:" not in res.cleaned
    assert "Oregon Department of Agriculture" not in res.cleaned
    assert (
        "Vaccines have undergone rigorous testing before approval."
        in res.cleaned
    )
    assert any(
        r.rule == "our_ruling" and r.mode == "section" for r in res.removals
    )


def test_our_ruling_our_rating_colon_removed():
    doc = (
        "The video was taken out of context from a 2019 rally. Our "
        "rating: False."
    )
    res = strip(doc, "politifact.com")
    assert "Our rating:" not in res.cleaned
    assert (
        "The video was taken out of context from a 2019 rally."
        in res.cleaned
    )


def test_our_ruling_prose_without_colon_unchanged():
    doc = "the ruling by the court was appealed the following week."
    res = strip(doc, "politifact.com")
    assert res.cleaned == doc
    assert res.removals == []


# --- our_verdict_colon now mode="sentence" (politifact + snopes) ----------


@pytest.mark.parametrize("domain", ["politifact.com", "snopes.com"])
def test_our_verdict_colon_sentence_mode(domain):
    doc = "Some evidence sentence. Our verdict: Mostly True."
    res = strip(doc, domain)
    assert "Our verdict" not in res.cleaned
    assert "Mostly True" not in res.cleaned
    assert "Some evidence sentence." in res.cleaned
    assert res.removals[0].mode == "sentence"
    assert res.removals[0].rule == "our_verdict_colon"


# --- dw.com "verdict_colon" mode="sentence" ---------------------------------


def test_dw_verdict_colon_sentence_mode():
    doc = (
        "The claim states that the vaccine contains microchips. "
        "Verdict: False. The explanation follows here: no chips were "
        "found in any sample tested."
    )
    res = strip(doc, "dw.com")
    assert "Verdict:" not in res.cleaned
    assert (
        "The claim states that the vaccine contains microchips."
        in res.cleaned
    )
    assert "no chips were found in any sample tested." in res.cleaned
    assert res.removals[0].mode == "sentence"
    assert res.removals[0].rule == "verdict_colon"


# --- indiatoday.in "crow_legend" (tagline) ----------------------------------


def test_crow_legend_removed_without_jhooth_tagline():
    # JHOOTH tagline absent; the crow legend must still fire on its own.
    doc = (
        "The purported image shows an elite commando of the "
        "paramilitary force. 1 Crow: Half True 2 Crows: Mostly lies "
        "3 Crows: Absolutely false"
    )
    res = strip(doc, "indiatoday.in")
    assert "Crow" not in res.cleaned
    assert "Absolutely false" not in res.cleaned
    assert (
        "The purported image shows an elite commando of the "
        "paramilitary force." in res.cleaned
    )
    assert res.removals[0].mode == "tagline"
    assert res.removals[0].rule == "crow_legend"


# --- FOOTER: boomlive.in "boomlive_adblock" (token) -------------------------


def test_boomlive_adblock_nag_removed_doc_becomes_empty():
    doc = (
        "Our website is made possible by displaying online "
        "advertisements to our visitors. Please consider supporting "
        "us by disabling your ad blocker. Please reload after ad "
        "blocker is disabled."
    )
    res = strip(doc, "boomlive.in")
    assert res.cleaned == ""
    assert any(
        r.rule == "boomlive_adblock" and r.category == "footer"
        for r in res.removals
    )


# --- FOOTER: snopes.com "snopes_meta_signoff" (sentence) --------------------


def test_snopes_meta_signoff_happy_reading_removed():
    doc = (
        "This article's methodology section is meant to help explain "
        "how we do what we do. Happy reading and let us know what "
        "else you might be interested in knowing. In February 2022, "
        "Facebook users shared a phishing scam."
    )
    res = strip(doc, "snopes.com")
    assert "Happy reading" not in res.cleaned
    assert (
        "help explain how we do what we do." in res.cleaned
    )
    assert (
        "In February 2022, Facebook users shared a phishing scam."
        in res.cleaned
    )
    assert any(
        r.rule == "snopes_meta_signoff" and r.category == "footer"
        for r in res.removals
    )


def test_snopes_meta_signoff_submit_ads_removed():
    doc = (
        "The claim originated from a Facebook post shared in 2021. "
        "Feel free to submit ads to us, and be sure to include a "
        "screenshot of the ad."
    )
    res = strip(doc, "snopes.com")
    assert "Feel free to submit ads to us" not in res.cleaned
    assert (
        "The claim originated from a Facebook post shared in 2021."
        in res.cleaned
    )


# --- FOOTER: dw.com "dw_translator_credit" (section) ------------------------


def test_dw_translator_credit_removed():
    doc = (
        "The republic was the result of the Ukrainian independence "
        "movement under the rule of the Russian czar. The "
        "territories that make up today's Ukraine were largely "
        "incorporated into the Russian Empire until its collapse in "
        "1917.\n\nThis article was adapted from German by Michael "
        "Trobridge"
    )
    res = strip(doc, "dw.com")
    assert "adapted from German" not in res.cleaned
    assert "Michael Trobridge" not in res.cleaned
    assert "until its collapse in 1917." in res.cleaned
    assert any(
        r.rule == "dw_translator_credit" and r.category == "footer"
        for r in res.removals
    )


# --- FOOTER: factly.in "factly_series_footer" (section) ---------------------


def test_factly_series_footer_removed():
    doc = (
        "The scheme provided direct cash transfers to farmers "
        "according to official records. This story is part of a "
        "larger series on the 4-years of the Modi government. This "
        "series has been made possible with the flash grant of the "
        "International Fact Checking Network (IFCN). Read the rest "
        "of the stories in this series here."
    )
    res = strip(doc, "factly.in")
    assert "This story is part of a larger series" not in res.cleaned
    assert "flash grant" not in res.cleaned
    assert "Read the rest of the stories" not in res.cleaned
    assert (
        "The scheme provided direct cash transfers to farmers "
        "according to official records." in res.cleaned
    )
    assert any(
        r.rule == "factly_series_footer" and r.category == "footer"
        for r in res.removals
    )


# --- FOOTER: africacheck.org fellowship byline + rate-overview -------------


def test_africacheck_fellowship_byline_and_rate_overview_removed():
    doc = (
        "The clinic denied administering unapproved vaccines to "
        "children. We therefore rate this claim as misleading. ( "
        "Read our overview of how we rate claims here . ) This "
        "report was written by Aria Deemie , a fact-checker and "
        "social media coordinator with Local Voices Liberia , during "
        "the 2023 Africa Check Fellowship in Nairobi, Kenya. The "
        "fellowship was sponsored by the US Department of State."
    )
    res = strip(doc, "africacheck.org")
    # The fellowship byline, rate-overview plug, and the existing
    # we_rate_claim verdict rule all fire; only the genuine leading
    # evidence sentence survives.
    assert res.cleaned == (
        "The clinic denied administering unapproved vaccines to "
        "children."
    )
    rules = {r.rule for r in res.removals}
    assert "africacheck_fellowship_byline" in rules
    assert "africacheck_rate_overview" in rules
    footer_rules = {
        r.rule for r in res.removals if r.category == "footer"
    }
    assert "africacheck_fellowship_byline" in footer_rules
    assert "africacheck_rate_overview" in footer_rules


def test_africacheck_fellowship_byline_negative_no_fellowship_mention():
    doc = (
        "The bridge was inaugurated in 2021 according to state "
        "media. This report was written by our correspondent in "
        "Lagos."
    )
    res = strip(doc, "africacheck.org")
    assert res.cleaned == doc
    assert not any(
        r.rule == "africacheck_fellowship_byline" for r in res.removals
    )


# --- FOOTER: leadstories.com "leadstories_wrote_about" (section) -----------


def test_leadstories_wrote_about_removed_to_eod():
    doc = (
        "The scam websites are designed to trick innocent people on "
        "the internet. We wrote about cbsnews.us before, here are "
        "our most recent articles that mention the site: Fake Photo "
        "Claims To Show Biden Asleep At Meeting; Old Video Falsely "
        "Linked To Recent Earthquake"
    )
    res = strip(doc, "leadstories.com")
    assert "We wrote about" not in res.cleaned
    assert "Fake Photo Claims To Show Biden" not in res.cleaned
    assert (
        "The scam websites are designed to trick innocent people on "
        "the internet." in res.cleaned
    )
    assert any(
        r.rule == "leadstories_wrote_about" and r.category == "footer"
        for r in res.removals
    )


# --- FOOTER: washingtonpost.com "wapo_factchecker_promos" (token) ----------


def test_wapo_factchecker_promos_all_removed_evidence_kept():
    doc = (
        "The senator cannot suggest that they reduced the budget "
        "deficit. Send us facts to check by filling out this form "
        "Sign up for The Fact Checker weekly newsletter The Fact "
        "Checker is a verified signatory to the International "
        "Fact-Checking Network code of principles Mar-a-Lago isn't "
        "just Trump's vacation spot."
    )
    res = strip(doc, "washingtonpost.com")
    assert "Send us facts to check" not in res.cleaned
    assert "Sign up for The Fact Checker" not in res.cleaned
    assert "verified signatory" not in res.cleaned
    assert (
        "The senator cannot suggest that they reduced the budget "
        "deficit." in res.cleaned
    )
    assert "Mar-a-Lago isn't just Trump's vacation spot." in res.cleaned
    assert all(
        r.mode == "token"
        for r in res.removals
        if r.rule == "wapo_factchecker_promos"
    )
    assert any(
        r.rule == "wapo_factchecker_promos" for r in res.removals
    )


# --- FOOTER: factcheckhub.com "factcheckhub_republished" (section) --------


def test_factcheckhub_republished_credit_removed():
    doc = (
        "The claims made in the viral post are INACCURATE.\n\nThis "
        "report was written by Daily Trust, and republished by The "
        "FactCheckHub as part of the Nigerian Fact-checkers' "
        "Coalition initiative. See the original piece here."
    )
    res = strip(doc, "factcheckhub.com")
    assert "This report was written by" not in res.cleaned
    assert "republished by The FactCheckHub" not in res.cleaned
    assert "See the original piece here." not in res.cleaned
    assert "The claims made in the viral post are INACCURATE." in (
        res.cleaned
    )
    assert any(
        r.rule == "factcheckhub_republished" and r.category == "footer"
        for r in res.removals
    )


# --- FOOTER: newschecker.in tipline meta-clause (token, bounded) ----------


def test_tipline_mention_removed_glued_periods_bounded():
    doc = (
        "The manifesto promised free electricity supply to mosques "
        "and churches.\"We also received this claim on our Whatsapp "
        "tipline (9999499044), too, requesting it to be "
        "fact-checked.Please see the congress manifesto."
    )
    res = strip(doc, "newschecker.in")
    assert "tipline" not in res.cleaned
    assert "9999499044" not in res.cleaned
    assert (
        "free electricity supply to mosques and churches." in res.cleaned
    )
    assert "Please see the congress manifesto." in res.cleaned
    assert any(
        r.rule == "newschecker_tipline" and r.mode == "token" and (
            r.category == "footer"
        )
        for r in res.removals
    )


def test_tipline_helpline_variant_removed():
    doc = (
        "Newschecker also received a request from a user on its "
        "WhatsApp helpline number (+91 9999499044), to check the "
        "veracity of this message. A keyword search found no such "
        "announcement."
    )
    res = strip(doc, "newschecker.in")
    assert "9999499044" not in res.cleaned
    assert "A keyword search found no such announcement." in res.cleaned


def test_tipline_caption_without_received_subject_survives():
    # Glued caption + evidence: the bounded token pattern must NOT fire
    # (sentence mode used to drag the keyword-search evidence away).
    doc = (
        "Screengrab of the request received in Newschecker's WhatsApp "
        "TiplineFactTo begin with, Newschecker did a keyword search "
        "and found that there were no reports."
    )
    res = strip(doc, "newschecker.in")
    assert "keyword search" in res.cleaned
    assert not any(
        r.rule == "newschecker_tipline" for r in res.removals
    )


def test_vishvas_claim_header_quote_survives():
    # vishvasnews has no tipline rule: its only tipline mention sits
    # inside the article's Claim-header quote and must survive.
    doc = (
        "Claim The post received by Vishvas News on its Whatsapp "
        "tipline reads: “Congratulations. You have been selected "
        "to benefit from the support provided by the World Health "
        "Organization for 500 Euros.”"
    )
    res = strip(doc, "vishvasnews.com")
    assert "Congratulations" in res.cleaned
    assert res.removals == []


# --- FOOTER: politifact.com "politifact_nc_project" (section) -------------


def test_politifact_nc_project_credit_removed():
    doc = (
        "Most residents in the county get their coverage from "
        "employers, Medicare or Medicaid. This story was produced "
        "by the North Carolina Fact-Checking Project, a partnership "
        "of McClatchy Carolinas, the Duke University Reporters' Lab "
        "and PolitiFact. The NC Local News Lab Fund and the "
        "International Fact-Checking Network provide support."
    )
    res = strip(doc, "politifact.com")
    assert "North Carolina Fact-Checking Project" not in res.cleaned
    assert "NC Local News Lab Fund" not in res.cleaned
    assert (
        "Most residents in the county get their coverage from "
        "employers, Medicare or Medicaid." in res.cleaned
    )
    assert any(
        r.rule == "politifact_nc_project" and r.category == "footer"
        for r in res.removals
    )


def test_dw_factcheck_label_all_occurrences_removed():
    doc = (
        "Claim: Kyiv seeks nuclear weapons. DW fact check: False. "
        "There is no evidence to date of such a program. "
        "Claim: The photo shows Zelenska. DW fact check: False/misleading. "
        "A reverse image search finds the pictured woman elsewhere."
    )
    res = strip(doc, "dw.com")
    assert "DW fact check" not in res.cleaned
    assert "no evidence to date" in res.cleaned
    assert "reverse image search" in res.cleaned
    labels = [
        r for r in res.removals if r.rule == "dw_factcheck_label"
    ]
    assert len(labels) == 2
