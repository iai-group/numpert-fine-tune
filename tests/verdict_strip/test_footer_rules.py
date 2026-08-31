"""Footer / sign-off stripping rules (precision-first)."""

import pytest

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.clean_splits import (  # noqa: E501
    clean_records,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.stripper import (  # noqa: E501
    Removal,
    strip,
)


def test_removal_category_defaults_to_verdict():
    r = Removal(rule="x", mode="section", span=(0, 1), removed_text="a")
    assert r.category == "verdict"


def test_footer_plumbing_inert_without_rules():
    # A domain with no footer rule and no verdict rule is unchanged.
    doc = "Some ordinary evidence sentence with no boilerplate."
    res = strip(doc, "example.com")
    assert res.cleaned == doc
    assert res.removals == []


# (domain, doc [real evidence lead-in + real footer], evidence_kept,
#  footer_gone)
SECTION_FIXTURES = [
    (
        "theferret.scot",
        "However, the campaign is not government-run but was started by a "
        "former police officer in 2013. Ferret Fact Service (FFS) is a "
        "non-partisan fact checker, working to the International "
        "Fact-Checking Network fact-checkers’ code of principles .",
        "former police officer in 2013.",
        "Ferret Fact Service (FFS) is",
    ),
    (
        "verafiles.org",
        "The photo actually dates from a 2018 rally. Have you seen any "
        "dubious claims, photos, memes, or online posts that you want us to "
        "verify? (Editor's Note: VERA Files has partnered with Facebook to "
        "fight the spread of disinformation. Find out more about this "
        "partnership and our methodology.)",
        "photo actually dates from a 2018 rally.",
        "Have you seen any dubious claims",
    ),
    (
        "factcheckni.org",
        "The next Assembly elections are due to be held in May 2022. "
        "FactCheckNI is Northern Ireland’s first and only dedicated "
        "independent fact-checking service and a verified signatory to the "
        "International Fact-Checking Network’s Code of Principles.",
        "due to be held in May 2022.",
        "FactCheckNI is Northern Ireland",
    ),
    (
        "altnews.in",
        "Such content must be cross-checked before putting it out in the "
        "public sphere. Donate to Alt News! Independent journalism that "
        "speaks truth to power and is free of corporate and political "
        "control is possible only when people contribute towards the same.",
        "public sphere.",
        "Donate to Alt News!",
    ),
    (
        "apnews.com",
        "The claim demonstrates the limitations of an autopsy by itself.___"
        "This is part of AP’s effort to address widely shared "
        "misinformation, including work with outside companies and "
        "organizations.Learn more about fact-checking at AP.",
        "limitations of an autopsy by itself.",
        "This is part of AP",
    ),
    (
        # NOTE: usatoday.com is in KEYWORD_DOMAINS (verdict "debunk"/"we rate"
        # rules also fire), so the evidence lead-in avoids those triggers.
        "usatoday.com",
        "The images were taken in 2019, long before the event.Thank you for "
        "supporting our journalism. You can subscribe to our print edition, "
        "ad-free app or e-newspaperhere.Our fact-check work is supported in "
        "part by a grant from Facebook.",
        "The images were taken in 2019, long before the event.",
        "Thank you for supporting our journalism",
    ),
    (
        "correctiv.org",
        "Die Behauptung ist damit falsch. Ihre Spende gegen Fake News "
        "Falschmeldungen und Verschwörungsmythen spalten die "
        "Gesellschaft. CORRECTIV.Faktencheckist eine eigenständige "
        "Redaktion des gemeinnützigen Recherchezentrums CORRECTIV.",
        "Die Behauptung ist damit falsch.",
        "Ihre Spende gegen Fake News",
    ),
    (
        "aap.com.au",
        "AAP FactCheck is an accredited member of the International "
        "Fact-Checking Network . To keep up with our latest fact checks, "
        "follow us on Facebook , Twitter and Instagram . All information, "
        "text and images included on the AAP Websites is for personal use "
        "only.",
        "accredited member of the International Fact-Checking Network",
        "To keep up with our latest fact checks, follow us on",
    ),
    (
        "newsmobile.in",
        "Therefore, we can conclusively say that the viral video's claim is "
        "false.If you want to fact-check any story, WhatsApp it now on +91 "
        "11 7127 9799FAKE NEWS BUSTER Name Email Phone",
        "the viral video's claim is false.",
        "If you want to fact-check any story",
    ),
]


@pytest.mark.parametrize(
    "domain,doc,evidence_kept,footer_gone",
    SECTION_FIXTURES,
    ids=[f[0] for f in SECTION_FIXTURES],
)
def test_section_footer_removed_evidence_kept(
    domain, doc, evidence_kept, footer_gone
):
    res = strip(doc, domain)
    assert footer_gone not in res.cleaned, "footer text should be removed"
    assert evidence_kept in res.cleaned, "genuine evidence must be preserved"
    assert any(
        r.category == "footer" for r in res.removals
    ), "a footer removal must be recorded"


def test_ifcn_factchat_one_liner_removed_midphrase():
    # Real politifact tail: verdict sentence, IFCN one-liner, then a stub.
    doc = (
        "He is the only major-party nominee in decades who has not done so. "
        "We rate Biden’s claim True. This fact check is available at "
        "IFCN’s 2020 US Elections FactChat #Chatbot on WhatsApp. "
        "Click here, for more."
    )
    res = strip(doc, "politifact.com")
    assert "This fact check is available at IFCN" not in res.cleaned
    # Preceding evidence sentence survives.
    assert "only major-party nominee in decades" in res.cleaned
    assert any(
        r.category == "footer" and r.rule == "ifcn_factchat"
        for r in res.removals
    )


def test_ifcn_rule_applies_across_domains():
    doc = (
        "The image is genuine. This fact check is available at IFCN's 2020 "
        "US Elections FactChat #Chatbot on WhatsApp."
    )
    res = strip(doc, "washingtonpost.com")
    assert "This fact check is available at IFCN" not in res.cleaned
    assert "The image is genuine." in res.cleaned


def test_footer_domain_without_footer_is_unchanged():
    # An altnews doc that happens to lack the donation footer is untouched.
    doc = "The viral video is from a 2019 protest, not the recent event."
    res = strip(doc, "altnews.in")
    assert res.cleaned == doc
    assert res.removals == []


def test_dw_com_content_not_stripped():
    # dw.com has NO footer rule; its repeated Ukraine content is real
    # evidence and must survive (guards the discovery's false positive).
    doc = (
        "The republic was the result of the Ukrainian independence movement "
        "under the rule of the Russian czar. The territories that make up "
        "today's Ukraine were largely incorporated into the Russian Empire "
        "until its collapse in 1917."
    )
    res = strip(doc, "dw.com")
    assert res.cleaned == doc
    assert res.removals == []


def test_clean_records_diff_log_has_category():
    records = [{
        "id": 1,
        "datasplit": "train",
        "url": "https://www.altnews.in/some-fact-check/",
        "doc": "The claim is baseless. Donate to Alt News! Independent "
               "journalism is possible only when people contribute.",
    }]
    cleaned, difflog = clean_records(records)
    assert "Donate to Alt News!" not in cleaned[0]["doc"]
    assert len(difflog) == 1
    cats = [rm["category"] for rm in difflog[0]["removals"]]
    assert "footer" in cats


def test_newschecker_space_variant_removes_full_contact_block():
    # 'fact check' with a SPACE (the bug: fact-?check missed this variant),
    # so the whole reader-contact block must be removed, not just the
    # trailing 'You can also visit' sentence.
    doc = (
        "The footage has been falsely linked to the 2023 Karnataka Assembly "
        "elections.Result: FalseSourcesTimes Now report, December 1, 2018"
        "If you would like us to fact check a claim, give feedback or lodge "
        "a complaint, WhatsApp us at 9999499044 or email us at "
        "[email protected] You can also visit the Contact Us page and fill "
        "the form."
    )
    res = strip(doc, "newschecker.in")
    assert "If you would like us to fact check a claim" not in res.cleaned
    assert "WhatsApp us at 9999499044" not in res.cleaned
    assert "2023 Karnataka Assembly elections." in res.cleaned
    assert any(r.category == "footer" for r in res.removals)


def test_checkyourfact_removes_full_signoff_incl_byline_and_dcnf():
    doc = (
        "Bannon’s claim that Obama spent a trillion dollars is true. "
        "Follow Kush on Twitter Have a fact check suggestion? Send ideas to "
        "[email protected] All content created by the Daily Caller News "
        "Foundation, an independent and nonpartisan newswire service, is "
        "available without charge to any legitimate news publisher that can "
        "provide a large audience. All republished articles must include our "
        "logo, our reporter’s byline and their DCNF affiliation."
    )
    res = strip(doc, "checkyourfact.com")
    assert "Follow Kush on Twitter" not in res.cleaned
    assert ("All content created by the Daily Caller News Foundation"
            not in res.cleaned)
    assert "available without charge" not in res.cleaned
    assert "All republished articles must include our logo" not in res.cleaned
    assert "spent a trillion dollars is true." in res.cleaned
    assert any(r.category == "footer" for r in res.removals)


def test_vishvas_removes_reader_request_cta():
    doc = (
        "The airstrike claim is false. Know the truth! If you have any "
        "doubts about any information or a rumor, do let us know! Knowing "
        "the truth is your right. If you feel any information is doubtful "
        "and it can impact the society or nation, send it to us by any of "
        "the sources mentioned below. Contribute to Vishwas News "
        "Fact-checking journalism. Help us in our fight against fake news. "
        "Tags"
    )
    res = strip(doc, "vishvasnews.com")
    assert "Know the truth!" not in res.cleaned
    assert ("send it to us by any of the sources mentioned below"
            not in res.cleaned)
    assert "Contribute to Vishwas News" not in res.cleaned
    assert "Tags" not in res.cleaned
    assert "The airstrike claim is false." in res.cleaned
    assert any(r.category == "footer" for r in res.removals)


def test_politifact_trailing_related_block_removed():
    doc = (
        "The cartoon ignores the security provisions in the bill. We rate "
        "this statement Mostly False. RELATED: Fact-checking misleading "
        "attacks on the HR 1 voting rights bill RELATED: No, HR 1 would not "
        "allow the Boston Marathon bomber to vote from prison RELATED: No, "
        "HR 1 doesn't 'allow minors to vote'"
    )
    res = strip(doc, "politifact.com")
    assert "RELATED:" not in res.cleaned
    assert ("The cartoon ignores the security provisions in the bill."
            in res.cleaned)
    assert any(r.category == "footer" and r.rule == "politifact_related"
               for r in res.removals)


def test_politifact_midarticle_related_preserved():
    # Mid-article RELATED (followed by substantial evidence to end-of-doc)
    # must NOT be removed — the anchor only matches a RELATED run that
    # reaches EOD, and >160 chars of evidence after a lone RELATED blocks it.
    doc = (
        "Public outcry followed the policy. RELATED: Mike Bloomberg's record "
        "on stop and frisk His campaign said the claim relied on data for the "
        "first quarter of 2012 and the last quarter of 2013. From January to "
        "March 2012, police made hundreds of thousands of stops across the "
        "city, according to department records reviewed by analysts."
    )
    res = strip(doc, "politifact.com")
    assert "His campaign said the claim relied on data" in res.cleaned
    assert "police made hundreds of thousands of stops" in res.cleaned
    assert not any(r.rule == "politifact_related" for r in res.removals)


def test_politifact_related_block_with_internal_period_fully_removed():
    # id4-style: a middle RELATED title contains a period; the WHOLE trailing
    # run must go, not just the final period-free link.
    doc = (
        "States have certified the 2020 election results. We rate this "
        "statement Pants on Fire. RELATED: GOP lawmaker’s proposal to "
        "decertify Arizona election won’t change anything about 2020 "
        "RELATED: A coalition of ‘stop the steal’ Republicans aims to take "
        "control of US elections. QAnon is helping RELATED: All of our "
        "fact-checks about elections"
    )
    res = strip(doc, "politifact.com")
    assert "RELATED:" not in res.cleaned
    assert "QAnon is helping" not in res.cleaned
    assert "States have certified the 2020 election results." in res.cleaned


def test_politifact_top_related_cluster_preserved():
    # RELATED links clustered near the TOP, followed by the article body,
    # must NOT be removed (only trailing runs are footers).
    doc = (
        "RELATED: Do red flag laws allow gun seizure without a judge’s OK? "
        "RELATED: Can you buy a gun after a felony? Anyone seeking to buy a "
        "handgun must fill out a DOJ form that asks about disqualifying "
        "conduct. They must disclose felony convictions. The claim is "
        "therefore missing context and needs clarification to be understood."
    )
    res = strip(doc, "politifact.com")
    assert ("Anyone seeking to buy a handgun must fill out a DOJ form"
            in res.cleaned)
    assert "must disclose felony convictions." in res.cleaned


def test_politifact_contributor_byline_removed():
    doc = (
        "The tax returns show he paid the reported amount. PolitiFact "
        "researcher Caryn Baird contributed to this report."
    )
    res = strip(doc, "politifact.com")
    assert "contributed to this report" not in res.cleaned
    assert ("The tax returns show he paid the reported amount."
            in res.cleaned)
    assert any(r.category == "footer" for r in res.removals)


def test_ifcn_factchat_removes_click_here_tail():
    doc = (
        "The video is authentic. This fact check is available at IFCN's "
        "2020 US Elections FactChat #Chatbot on WhatsApp. Click here, for "
        "more."
    )
    res = strip(doc, "politifact.com")
    assert "IFCN" not in res.cleaned
    assert "Click here, for more." not in res.cleaned
    assert "The video is authentic." in res.cleaned
    assert any(r.category == "footer" and r.rule == "ifcn_factchat"
               for r in res.removals)


def test_ifcn_factchat_removes_glued_clickhere_tail():
    doc = (
        "The claim is baseless. This fact check is available at IFCN’s 2020 "
        "US Elections FactChat #Chatbot on WhatsApp. Clickhere, for more."
    )
    res = strip(doc, "washingtonpost.com")
    assert "Clickhere" not in res.cleaned
    assert "IFCN" not in res.cleaned
    assert "The claim is baseless." in res.cleaned


def test_ifcn_factchat_without_click_tail_still_removed():
    doc = (
        "The image is genuine. This fact check is available at IFCN's 2020 "
        "US Elections FactChat #Chatbot on WhatsApp."
    )
    res = strip(doc, "healthfeedback.org")
    assert "IFCN" not in res.cleaned
    assert "The image is genuine." in res.cleaned


def test_politifact_ratings_legend_removed():
    doc = (
        "We traced the figure to a lobbying group. We rate Righter's claim "
        "False. FALSE — The statement is not accurate. Click here for more "
        "on the six PolitiFact ratings and how we select facts to check."
    )
    res = strip(doc, "politifact.com")
    assert "FALSE — The statement is not accurate." not in res.cleaned
    assert ("Click here for more on the six PolitiFact ratings"
            not in res.cleaned)
    assert "We traced the figure to a lobbying group." in res.cleaned
    assert any(r.category == "footer"
               and r.rule == "politifact_ratings_legend"
               for r in res.removals)


def test_politifact_ratings_legend_is_case_sensitive_not_prose():
    # lowercase "false — the statement" in evidence prose must NOT trigger.
    doc = (
        "The mayor said it was false — the statement he referred to was "
        "later corrected by his office and clarified in a press release."
    )
    res = strip(doc, "politifact.com")
    assert ("the statement he referred to was later corrected"
            in res.cleaned)
    assert not any(r.rule == "politifact_ratings_legend"
                   for r in res.removals)


def test_politifact_ratings_legend_without_click_tail_removed():
    # Legend definition line with NO "Click here" tail must still go.
    doc = (
        "Experts disagreed with the figure cited by the senator. FALSE – "
        "The statement is not accurate."
    )
    res = strip(doc, "politifact.com")
    assert "FALSE – The statement is not accurate." not in res.cleaned
    assert ("Experts disagreed with the figure cited by the senator."
            in res.cleaned)
    assert any(r.rule == "politifact_ratings_legend" for r in res.removals)


def test_politifact_ratings_legend_glued_tail_removed():
    # Missing spaces ("formoreon", dash glued to the rating) must still match.
    doc = (
        "The data does not support the figure. MOSTLY TRUE– The statement "
        "is accurate but needs clarification or additional information. "
        "Click here formoreon the six PolitiFact ratings and how we select "
        "facts to check."
    )
    res = strip(doc, "politifact.com")
    assert ("The statement is accurate but needs clarification"
            not in res.cleaned)
    assert "formoreon" not in res.cleaned
    assert "The data does not support the figure." in res.cleaned
    assert any(r.rule == "politifact_ratings_legend" for r in res.removals)


# --- deferred-pattern pass (2026-07-02): boomlive / sharethefacts /
# --- leadstories / IFCN plural variant / factcheckhub bio ------------------


def test_boomlive_syndication_notice_removed():
    # Real tail (train 248): syndication disclaimer in parentheses at EOD.
    # (Lead-in must not contain verdict keywords like "debunked" — the
    # boomlive verdict rule removes those sentences separately, by design.)
    doc = (
        "The count included 35.9 million early voters, and 65.2 million "
        "votes by mail, according to US Elections project estimates. "
        "(Except for the headline, this story has not been edited by BOOM "
        "staff and is published from a syndicated feed.)"
    )
    res = strip(doc, "boomlive.in")
    assert "US Elections project estimates." in res.cleaned
    assert "Except for the headline" not in res.cleaned
    assert any(r.rule == "boomlive_syndication" for r in res.removals)
    assert all(
        r.category == "footer"
        for r in res.removals if r.rule == "boomlive_syndication"
    )


def test_sharethefacts_url_removed_globally():
    # Real politifact tail (train 115 shape): bare share-widget URL. The
    # rule is global token-mode, so it must also fire off-domain (a
    # boomlive doc carries one too).
    url = (
        "https://www.sharethefacts.co/share/"
        "02e64cbe-e4c0-43d0-8a3d-3350ccfecf97"
    )
    doc = f"The senator's figure matches the census data. {url}"
    for domain in ("politifact.com", "boomlive.in"):
        res = strip(doc, domain)
        assert "matches the census data." in res.cleaned
        assert "sharethefacts.co" not in res.cleaned
        assert any(r.rule == "sharethefacts_url" for r in res.removals)


def test_leadstories_related_sentence_removed():
    # Real tail (train 2807): self-promotional related-fact-checks line,
    # incl. the multi-link "here, here and here" form.
    doc = (
        "Should we receive a response, this article will be updated as "
        "appropriate. Other Lead Stories fact checks about Ronaldo can be "
        "found here, here and here."
    )
    res = strip(doc, "leadstories.com")
    assert "updated as appropriate." in res.cleaned
    assert "Other Lead Stories" not in res.cleaned


def test_leadstories_related_glued_to_source_caption_preserves_caption():
    # Real doc shape (train 9908): the footer line is glued to a
    # "(Source: ...)" screenshot caption with no sentence boundary PySBD
    # can split — token mode must excise only the footer, not the caption.
    doc = (
        "A longer version of this video was posted on YouTube in December "
        "2019, and the caption described it as a scene recorded in Syria:"
        "(Source: YouTube screenshot taken on Mon Aug 21 14:05:03 2023 "
        "UTC)Other Lead Stories fact checks about the war in Ukraine can "
        "be found here."
    )
    res = strip(doc, "leadstories.com")
    assert "(Source: YouTube screenshot" in res.cleaned
    assert "posted on YouTube in December 2019" in res.cleaned
    assert "Other Lead Stories" not in res.cleaned


def test_leadstories_another_variant_removed():
    # Real doc shape (train 4591): "Another Lead Stories fact check ...".
    doc = (
        "The side-by-side images were found in a post on Twitter:(Source: "
        "Twitter screenshot taken on Tue Jun 13 22:06:14 2023 UTC)Another "
        "Lead Stories fact check of claims related to Judge Aileen Cannon "
        "can be found here."
    )
    res = strip(doc, "leadstories.com")
    assert "(Source: Twitter screenshot" in res.cleaned
    assert "Another Lead Stories" not in res.cleaned


def test_leadstories_evidence_found_here_preserved():
    # Real evidence (train 798): a source link that happens to end
    # "can be found here." must NOT be clipped (no "Other Lead Stories"
    # anchor).
    doc = (
        "ESA's space situational awareness page, which includes a risk "
        "list of potential asteroid threats, can be found here. We don't "
        "recommend the Alien Blog for information in life and death issues."
    )
    res = strip(doc, "leadstories.com")
    assert res.cleaned == doc


def test_ifcn_factchat_plural_variant_removed():
    # Real tail (val 319): "These fact checks are..." plural form of the
    # FactChat sign-off, with the Click-here tail.
    doc = (
        "Do masks cause carbon dioxide poisoning? These fact checks are "
        "available at IFCN's 2020 U.S. Elections FactChat #Chatbot on "
        "WhatsApp. Click here for more."
    )
    res = strip(doc, "example.com")
    assert "carbon dioxide poisoning?" in res.cleaned
    assert "FactChat" not in res.cleaned
    assert "Click here" not in res.cleaned


def test_factcheckhub_author_bio_sentences_removed():
    # Real tails (train 1235 / train 1854): reach-the-author + author-bio
    # sign-off sentences.
    doc = (
        "The claim that the video shows soldiers in Marte is FALSE. "
        "You can reach him via nyahaya@icirnigeria.org and "
        "@NurudeenAkewus1 via Twitter."
    )
    res = strip(doc, "factcheckhub.com")
    assert "is FALSE." in res.cleaned
    assert "You can reach him" not in res.cleaned

    doc2 = (
        "The video is from a 2018 military exercise. Opeyemi Kehinde is a "
        "Professional Fact-Checker, Multimedia Journalist, Strategic "
        "Content Curator and Editor of the FactCheckHub."
    )
    res2 = strip(doc2, "factcheckhub.com")
    assert "2018 military exercise." in res2.cleaned
    assert "Professional Fact-Checker" not in res2.cleaned


def test_clean_records_binary_schema_uses_claim_id():
    # binary_data records key on claim_id (no id/datasplit); the diff log
    # must still carry a usable identifier.
    records = [{
        "claim_id": "test_000123",
        "url": "https://www.altnews.in/some-fact-check/",
        "doc": "The claim is baseless. Donate to Alt News! Independent "
               "journalism is possible only when people contribute.",
    }]
    cleaned, difflog = clean_records(records)
    assert "Donate to Alt News!" not in cleaned[0]["doc"]
    assert difflog[0]["id"] == "test_000123"


def test_sharethefacts_url_spaced_normalization_removed():
    # binary_data's normalization inserts spaces after periods, mangling
    # the URL to "www. sharethefacts. co/share/..." — still boilerplate.
    doc = ("We rate this claim False. https://www. sharethefacts. "
           "co/share/16053b1b-0a59-4f24-ac3f-dcb041744f65")
    res = strip(doc, "politifact.com")
    assert "sharethefacts" not in res.cleaned


# --- final-pass rules (2026-07-03): WaPo widgets / logically disclaimer ----


def test_wapo_comment_gift_share_widgets_removed():
    # Real shapes: leading "Comment on this story Comment Gift Article
    # Share" UI junk and the trailing "GiftOutline Gift Article".
    doc = (
        "Comment on this story Comment Gift Article Share The fifth "
        "Democratic presidential debate of the 2020 campaign drew 6.5 "
        "million viewers. The Washington Post is a verified signatory to "
        "the International Fact-Checking Network code of principles "
        "GiftOutline Gift Article"
    )
    res = strip(doc, "washingtonpost.com")
    assert "drew 6.5 million viewers." in res.cleaned
    assert "Comment on this story" not in res.cleaned
    assert "GiftOutline" not in res.cleaned
    assert "Gift Article" not in res.cleaned


def test_wapo_share_the_facts_trailer_removed():
    # Real tail (train 344): share-widget metadata dump at EOD.
    doc = (
        "The tax cut applied to 82 percent of households. "
        "Share the Facts 2018-02-02 23:09:01 UTC Washington Post -1 -1 -1"
    )
    res = strip(doc, "washingtonpost.com")
    assert "82 percent of households." in res.cleaned
    assert "Share the Facts" not in res.cleaned
    assert "-1 -1 -1" not in res.cleaned


def test_logically_covid_disclaimer_removed_midtail():
    # Real shape (149 logically.ai docs): a stereotyped two-sentence
    # advisory block, sometimes followed by more content — token mode.
    doc = (
        "The image shows an event from 2019, not the vaccination drive. "
        "The COVID-19 pandemic has given rise to a lot of potentially "
        "dangerous misinformation. For reliable advice on COVID-19 "
        "including symptoms, prevention and available treatment, please "
        "refer to the World Health Organisation or your national "
        "healthcare authority. Have a question or correction on one of "
        "our fact-checks?"
    )
    res = strip(doc, "logically.ai")
    assert "an event from 2019" in res.cleaned
    assert "potentially dangerous misinformation" not in res.cleaned
    assert "World Health Organisation" not in res.cleaned
    # Content after the disclaimer survives (token, not section).
    assert "Have a question or correction" in res.cleaned


def test_logically_covid_disclaimer_organization_spelling():
    doc = (
        "The claim is missing context about the trial size of 43,000. The "
        "COVID-19 pandemic has given rise to a lot of potentially "
        "dangerous misinformation. For reliable advice on COVID-19 "
        "including symptoms, prevention and available treatment, please "
        "refer to the World Health Organization or your national "
        "healthcare authority."
    )
    res = strip(doc, "logically.ai")
    assert "trial size of 43,000." in res.cleaned
    assert "World Health Organization" not in res.cleaned


def test_wapo_glued_gift_share_and_share_bar_removed():
    # Real doc (val 1160 / train 2701 shapes): "Gift Article Share" glued
    # to the next word, plus the social share-bar widget run.
    doc = (
        "Gift Article ShareWhat Donald Trump is doing on the campaign "
        "trail Share Share Share on Facebook Share on Twitter Share on "
        "Reddit Share on LinkedIn MailSolid Email this link The claim "
        "cites 4.2 percent growth."
    )
    res = strip(doc, "washingtonpost.com")
    assert "cites 4.2 percent growth." in res.cleaned
    assert "Gift Article" not in res.cleaned
    assert "Share on Facebook" not in res.cleaned
    assert "MailSolid" not in res.cleaned


def test_logically_disclaimer_short_advice_variant():
    # Real variant (train 7003): "For reliable advice on symptoms, ..."
    # (no "COVID-19 including").
    doc = (
        "The funding it may have received in connection with COVID-19 was "
        "$3.9 million. The COVID-19 pandemic has given rise to a lot of "
        "potentially dangerous misinformation. For reliable advice on "
        "symptoms, prevention and available treatment, please refer to "
        "the World Health Organisation or your national healthcare "
        "authority."
    )
    res = strip(doc, "logically.ai")
    assert "$3.9 million." in res.cleaned
    assert "dangerous misinformation" not in res.cleaned
