"""Domain-anchored verdict-removal rules (data only).

Each rule fires ONLY on its domain and ONLY when its anchor regex matches.
Modes (see stripper.py for behavior):
  - tagline   : remove the matched span (fixed boilerplate string).
  - section   : remove from the anchor to the end of the doc.
  - sentence  : remove the whole sentence(s) containing the anchor.
  - token     : remove just the matched inline marker (glue-repair handled by
                the engine collapsing the resulting whitespace). Used for
                structural header labels whose following text is the evidence
                body (e.g. africacheck `Verdict incorrect`, checkyourfact
                `Verdict: X`).
  - label_next: remove a `<rating>:` header label AND the sentence after the
                colon (the feedback-sites `DETAILS <rating>: <explanation>`).
Anchors are derived from the empirical discovery in the design spec (§2).
"""

import re
from dataclasses import dataclass
from typing import Pattern
from urllib.parse import urlparse

MODES = ("tagline", "section", "sentence", "token", "label_next")


@dataclass(frozen=True)
class Rule:
    name: str
    anchor: Pattern
    mode: str


def domain_of(url: str) -> str:
    """Return the registrable host of `url`, lowercased, without `www.`."""
    if not url:
        return ""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


_I = re.IGNORECASE
_IS = re.IGNORECASE | re.DOTALL

# The science/health/climate-feedback trio shares a template: the same
# all-caps `KEY TAKE AWAY` takeaway header (207 docs) and `VERDICT SOURCE:
# <who>, <outlet>, <date> <year>` metadata line (207 docs) appear on all
# three (2026-07-08 discovery). Anchored here so both rules reuse one list.
_FEEDBACK_DOMAINS = (
    "healthfeedback.org", "sciencefeedback.co", "climatefeedback.org",
)

# Structural "Fact Check:" section header seen on six fact-checker domains
# (checkyourfact 287, vishvasnews 93, indiatoday 40, newsmeter 8, snopes 5,
# boomlive 3). The following text is the evidence body (or a related-link /
# citation title), so only the header token is removed. On vishvasnews the
# rule is belt-and-braces behind claim_review_block (footer, runs first and
# consumes ~all occurrences inside the trailing block); its only production
# firings there are mid-doc headers (e.g. train:818), where any following
# verdict-sentence residue is left to the LLM locate+gate pass.
_FACT_CHECK_DOMAINS = (
    "checkyourfact.com", "vishvasnews.com", "indiatoday.in",
    "newsmeter.in", "snopes.com", "boomlive.in",
)

# Rating words seen after "Verdict"/"the claim is"/"we rate" in discovery.
_RATING = (
    r"(?:mostly\s+)?(?:correct|incorrect|true|false|unproven|misleading|"
    r"checked|exaggerated|understated|overstated|distorted|downplayed|"
    r"mixture|partly\s+true|pants\s+on\s+fire|half\s+true)"
)

# Rating labels used by the science/health/climate-feedback sites, e.g.
# "DETAILS Misleading:" / "Flawed reasoning:" / "Inadequate support:".
# "Cherry-picking" is case-sensitive and hyphenated ((?-i:...) scopes out
# the rule's IGNORECASE): all 7 corpus label hits are "Cherry-picking:",
# while lowercase "cherry picking:" occurs as quoted PROSE (climatefeedback
# id=6510 "an example of cherry picking: selectively picking ...") and must
# survive. The other forms appear with a colon ONLY as rating labels, and
# only on the feedback trio (verified corpus-wide 2026-07-08).
_FEEDBACK_LABEL = (
    r"(?:(?:factually\s+)?(?:inaccurate|incorrect)|misleading|unsupported|"
    r"imprecise|inadequate support|flawed reasoning|misrepresents sources?|"
    r"misrepresents a complex reality|"
    r"fails to grasp significance of observation|"
    r"overstates scientific confidence|(?-i:Cherry-picking)|"
    r"lacks context|lack of context|mostly accurate|accurate|correct|"
    r"biased|overstated|exaggerated)"
)

# Cross-reference verdict markers ("we debunked this", "a similar false
# claim"). Domain-anchored (KEYWORD_DOMAINS) because "debunked" also appears in
# legitimate evidence on news/source sites; we only strip it on confirmed
# fact-checker domains and lean on the LLM audit to catch over-removal.
KEYWORD_DOMAINS = frozenset({
    "politifact.com", "snopes.com", "factcheck.afp.com", "boomlive.in",
    "usatoday.com", "verafiles.org", "leadstories.com", "factly.in",
    "reuters.com", "indiatoday.in", "aap.com.au", "africacheck.org",
    "checkyourfact.com", "fullfact.org", "healthfeedback.org",
    "sciencefeedback.co", "climatefeedback.org", "logically.ai",
})
KEYWORD_RULES = [
    Rule(
        "debunked",
        re.compile(r"[^.!?]*\bdebunk\w*\b[^.!?]*[.!?]", _I),
        "sentence",
    ),
    Rule(
        "similar_false_claim",
        re.compile(r"[^.!?]*\bsimilar false claims?\b[^.!?]*[.!?]", _I),
        "sentence",
    ),
]

RULES = {
    "indiatoday.in": [
        Rule(
            "jhooth_tagline",
            re.compile(
                r"\s*JHOOTH BOLE KAUVA KAATE.*?intensity of the lie\.?", _IS
            ),
            "tagline",
        ),
        # Crow-rating legend that follows (or appears without) the JHOOTH
        # tagline: "1 Crow: Half True 2 Crows: Mostly lies 3 Crows:
        # Absolutely false".
        Rule(
            "crow_legend",
            re.compile(
                r"(?:\s*\d\s*Crows?\s*:\s*"
                r"(?:Half\s+True|Mostly\s+lies|Absolutely\s+false)\.?){1,3}",
                _I,
            ),
            "tagline",
        ),
    ],
    "apnews.com": [
        # "AP'S ASSESSMENT: False." / "Missing context." / "Altered video."
        # The rating phrase is short (<=40 chars corpus-wide); a bounded
        # token match survives the glued "officers.AP'S ASSESSMENT: False.
        # Social..." docs where PySBD cannot split (label_next would
        # over-remove there).
        Rule(
            "ap_assessment",
            re.compile(r"AP.{0,2}S ASSESSMENT\s*:\s*[^.!?]{0,40}[.!?]?", _I),
            "token",
        ),
        # Structural section header; the text after it is the evidence body.
        # Case-sensitive: the corpus form is all-caps "THE FACTS:".
        Rule("the_facts_header", re.compile(r"\bTHE FACTS\s*:"), "token"),
    ],
    "politifact.com": [
        # Trailing "Our verdict: Mostly True." label. sentence mode per the
        # whole-sentence policy (the host sentence is just the label itself).
        Rule(
            "our_verdict_colon",
            re.compile(r"\bOur verdict\s*:\s*" + _RATING + r"\.?", _I),
            "sentence",
        ),
        # "The ruling:" / "Our rating:" open the same trailing verdict
        # section as "Our ruling"; the colon is required on the "The"/"rating"
        # variants so ordinary prose ("the ruling by the court") never fires.
        Rule(
            "our_ruling",
            re.compile(
                r"\bOur ?ruling\b|\b(?:The ruling|Our rating)\s*:", _I
            ),
            "section",
        ),
        # Structural "THE FACTS:" header (evidence body follows).
        # Case-sensitive all-caps, as on apnews.
        Rule("the_facts_header", re.compile(r"\bTHE FACTS\s*:"), "token"),
        # "We rate it/this/Perry's statement <rating>" — require a rating word
        # so non-verdict "we rate ..." prose is left alone.
        Rule(
            "we_rate",
            re.compile(
                r"\bWe rate\b[^.!?]*" + _RATING + r"[^.!?]*[.!?]", _I
            ),
            "sentence",
        ),
        # "We award Huckabee a Barely True"
        Rule(
            "we_award",
            re.compile(
                r"\bWe award\b[^.!?]*" + _RATING + r"[^.!?]*[.!?]", _I
            ),
            "sentence",
        ),
        # standalone "The claim is False." (no "Our ruling"/"We rate")
        Rule(
            "claim_is",
            re.compile(
                r"[^.!?]*\b(?:the|this) claim is "
                + _RATING
                + r"[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "theferret.scot": [
        # Mid-doc verdict header: "Ferret Fact Service verdict: True
        # <verdict explanation sentence>".
        Rule(
            "ferret_verdict",
            re.compile(
                r"Ferret Fact Service verdict\s*:?\s*" + _RATING, _I
            ),
            "label_next",
        ),
    ],
    "ghanafact.com": [
        # Metadata block "Source: <who> Verdict: FALSE Researched <name>".
        # token only (exempt from the whole-sentence policy): the metadata
        # block is glued to the article opening with no punctuation, so the
        # PySBD "sentence" spans ~300 chars of real evidence (measured).
        Rule(
            "verdict_colon",
            re.compile(r"\bVerdict\s*:\s*" + _RATING, _I),
            "token",
        ),
    ],
    "dw.com": [
        # Bare "Verdict: False." line between sub-claim and explanation in
        # dw's multi-fact-check compilations. sentence mode per the
        # whole-sentence policy (host sentence is the bare label; measured
        # extra removal vs token is ~2 chars corpus-wide).
        Rule(
            "verdict_colon",
            re.compile(r"\bVerdict\s*:\s*" + _RATING + r"\.?", _I),
            "sentence",
        ),
        # Compilation-header label "DW fact check: False[/misleading]",
        # repeated once per sub-claim (dozens per ~19k-char doc). The LLM
        # locate+gate only excises the FIRST occurrence of each located
        # quote string, so 19 hybrid docs kept later occurrences (found
        # 2026-07-08); a rule removes ALL matches deterministically.
        Rule(
            "dw_factcheck_label",
            re.compile(
                r"\bDW fact check\s*:\s*" + _RATING
                + r"(?:\s*/\s*\w+)?\.?",
                _I,
            ),
            "sentence",
        ),
    ],
    "snopes.com": [
        # Trailing "Our verdict: Mostly True." label. sentence mode per the
        # whole-sentence policy.
        Rule(
            "our_verdict_colon",
            re.compile(r"\bOur verdict\s*:\s*" + _RATING + r"\.?", _I),
            "sentence",
        ),
        # allow inserted adverbs: "we thus rate this claim as ..."
        Rule(
            "we_rate",
            re.compile(
                r"[^.!?]*\bwe (?:\w+\s){0,2}rate this claim[^.!?]*[.!?]", _I
            ),
            "sentence",
        ),
        # "We rank the claims ... as false"
        Rule(
            "we_rank",
            re.compile(
                r"[^.!?]*\bwe rank\b[^.!?]*" + _RATING + r"[^.!?]*[.!?]", _I
            ),
            "sentence",
        ),
    ],
    "africacheck.org": [
        # token only (exempt from the whole-sentence policy): "Verdict
        # incorrect Africa Check asked ..." is glued to following evidence
        # with no punctuation, so the PySBD "sentence" carries a median 141
        # (max ~1900) chars of real evidence (measured corpus-wide).
        Rule(
            "verdict_token",
            re.compile(r"\bVerdict\s*" + _RATING, _I),
            "token",
        ),
        # Colon form in multi-claim reports: "Verdict: Correct <verdict
        # restatement sentence>" — the restatement asserts the sub-claim's
        # veracity, so label_next takes the label AND that sentence.
        Rule(
            "verdict_colon",
            re.compile(r"\bVerdict\s*:\s*" + _RATING, _I),
            "label_next",
        ),
        # sentence form: "We therefore rate this claim as misleading."
        Rule(
            "we_rate_claim",
            re.compile(
                r"[^.!?]*\bwe (?:\w+\s){0,2}rate this claim[^.!?]*[.!?]", _I
            ),
            "sentence",
        ),
    ],
    "factcheck.afp.com": [
        Rule(
            "claim_is",
            re.compile(
                r"[^.!?]*\b(?:the|this) claim is "
                + _RATING
                + r"[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "factly.in": [
        Rule(
            "hence_claim",
            re.compile(
                r"[^.!?]*\b(?:the|this) claim is "
                + _RATING
                + r"[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "thequint.com": [
        # "Is it true?: The claim is false."
        Rule(
            "claim_is",
            re.compile(
                r"[^.!?]*\b(?:the|this) claim is "
                + _RATING
                + r"[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "logically.ai": [
        Rule(
            "the_verdict",
            re.compile(r"\bThe ?[Vv]erdict\b:?", _I),
            "section",
        ),
    ],
    "checkyourfact.com": [
        # sentence mode per the whole-sentence policy (host sentence is the
        # bare "Verdict: False." line; measured extra removal ~2 chars).
        Rule(
            "verdict_line",
            re.compile(r"\bVerdict:\s*" + _RATING, _I),
            "sentence",
        ),
    ],
    "washingtonpost.com": [
        Rule(
            "pinocchio_test",
            re.compile(r"\bThe Pinocchio Test\b", _I),
            "section",
        ),
        Rule(
            "wapo_rating",
            re.compile(r"\bWashington Post Rating:[^\n]*", _I),
            "token",
        ),
        Rule(
            "n_pinocchios",
            re.compile(
                r"[^.!?]*\b(?:one|two|three|four|\d)[ -]pinocchios?\b"
                r"[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "usatoday.com": [
        Rule(
            "we_rate",
            re.compile(
                r"[^.!?]*\b(?:based on our research,?\s*)?we rate "
                r"(?:true|false|[A-Z]{3,})[^.!?]*[.!?]",
                _I,
            ),
            "sentence",
        ),
    ],
    "healthfeedback.org": [
        Rule(
            "details_rating",
            re.compile(r"(?:\bDETAILS\s+)?\b" + _FEEDBACK_LABEL + r"\s*:", _I),
            "label_next",
        ),
    ],
    "sciencefeedback.co": [
        Rule(
            "details_rating",
            re.compile(r"(?:\bDETAILS\s+)?\b" + _FEEDBACK_LABEL + r"\s*:", _I),
            "label_next",
        ),
    ],
    "climatefeedback.org": [
        Rule(
            "details_rating",
            re.compile(r"(?:\bDETAILS\s+)?\b" + _FEEDBACK_LABEL + r"\s*:", _I),
            "label_next",
        ),
        # "REVIEW CLAIM:" restates the reviewed claim. token (header only):
        # the text after the colon duplicates the record's claim field (not
        # a verdict leak), and label_next over-removes into the adjacent
        # scientist-attribution sentence where PySBD merges the two.
        Rule("review_claim", re.compile(r"\bREVIEW CLAIM\s*:"), "token"),
    ],
    "aap.com.au": [
        Rule("the_verdict", re.compile(r"\bThe Verdict\b", _I), "section"),
    ],
    "boomlive.in": [
        # Headline-suffix marker "...quote? A FactCheck" plus the adjacent
        # bare all-caps "FACT-CHECK" section header (rule total: 118 docs /
        # 123 occurrences; the bare header is always followed by "BOOM
        # found ..."/"We ..."/"A reverse ...", never a lowercase-continued
        # word). token, case-sensitive so prose "a fact check" survives.
        # The negative lookahead keeps a hypothetical "FACT-CHECKing"
        # intact while allowing the glued uppercase org name
        # ("FACT-CHECKBOOM").
        Rule(
            "a_factcheck_header",
            re.compile(r"A FactCheck|FACT-?CHECK(?![a-z])"),
            "token",
        ),
    ],
}

# Structural "Fact Check:" header on the six fact-checker domains. Colon
# required and case-sensitive (`Fact Check\s*:`, tolerating vishvasnews's
# "Fact Check :" space) so ordinary prose "fact check" survives. token:
# remove the header only; the following text (evidence body or link/citation
# title) is left in place. Fires on every occurrence (indiatoday repeats it
# in "Also Read" blocks).
for _dom in _FACT_CHECK_DOMAINS:
    RULES.setdefault(_dom, []).append(
        Rule("fact_check_header", re.compile(r"\bFact Check\s*:"), "token")
    )

# Feedback-trio shared header/metadata rules (see _FEEDBACK_DOMAINS). All
# token, case-sensitive all-caps.
#   key_take_away  : "KEY TAKE AWAY" takeaway header (tolerating the
#                    "KEY TAKEAWAYS" variant) — a structural label whose
#                    following summary is evidence; only the header goes.
#   verdict_source : the "VERDICT SOURCE: <who>, <outlet>, <date> <year>"
#                    attribution line (207 docs, all 3 domains). Removed from
#                    the label through the FIRST 4-digit year (lazy, <=160
#                    chars): the year is not followed by reliable punctuation
#                    ("... 2021 DETAILS ..."), and verified corpus-wide the
#                    first year is always the publication date (never an
#                    earlier source-name digit). Docs with no year in reach
#                    are left unmatched (none exist in the corpus, but the
#                    guard is intentional). The trailing "DETAILS <rating>:"
#                    is handled by the existing details_rating rule.
for _dom in _FEEDBACK_DOMAINS:
    RULES.setdefault(_dom, []).extend([
        Rule("key_take_away", re.compile(r"\bKEY TAKE ?AWAYS?\b"), "token"),
        Rule(
            "verdict_source",
            re.compile(r"\bVERDICT SOURCE\s*:[\s\S]{0,160}?(?:19|20)\d{2}"),
            "token",
        ),
    ])

# Footer / sign-off boilerplate. Same Rule dataclass + modes as verdict rules.
# GLOBAL rules apply to every doc; FOOTER_RULES apply only on their domain.
GLOBAL_FOOTER_RULES: list[Rule] = [
    # Multi-domain FactChat sign-off + its optional "Click here for more"
    # tail. token mode removes exactly the matched span (works mid-document).
    # "U.S." must not terminate the sentence match (its periods otherwise
    # stop [^.!?]* early and leave a mangled "S. Elections ..." residue).
    Rule(
        "ifcn_factchat",
        re.compile(
            r"Th(?:is|ese)\s+fact\s+checks?\s+(?:is|are)\s+available\s+at"
            r"\s+IFCN(?:U\.S\.|[^.!?])*[.!?]"
            r"(?:\s*Click\s*here,?\s*for more[^.!?]*[.!?]?)?",
            _I,
        ),
        "token",
    ),
    # sharethefacts.co share-widget URLs (politifact, one boomlive doc).
    # A bare URL carries no evidence, so token removal is safe anywhere.
    # \s* after the dots: binary_data's normalization inserts spaces
    # ("www. sharethefacts. co/...").
    Rule(
        "sharethefacts_url",
        re.compile(
            r"https?://(?:www\.\s*)?sharethefacts\.\s*co/share/\S+", _I
        ),
        "token",
    ),
]

FOOTER_RULES: dict[str, list[Rule]] = {
    "theferret.scot": [
        Rule("ferret_signoff",
             re.compile(r"Ferret Fact Service \(FFS\) is", _I), "section"),
    ],
    "verafiles.org": [
        Rule("vera_signoff",
             re.compile(
                 r"Have you seen any dubious claims"
                 r"|Find out more about this partnership and our methodology"
                 r"|\(Editor.s Note: VERA Files has partnered", _I),
             "section"),
    ],
    "factcheckni.org": [
        Rule("factcheckni_signoff",
             re.compile(r"FactCheckNI is Northern Ireland.s first", _I),
             "section"),
    ],
    "altnews.in": [
        Rule("altnews_donate",
             re.compile(r"Donate to Alt News!", _I), "section"),
    ],
    "apnews.com": [
        Rule("ap_effort",
             re.compile(
                 r"_{2,}\s*This is part of AP.s effort to address"
                 r"|Learn more about fact-?checking at AP\.", _I),
             "section"),
    ],
    "usatoday.com": [
        Rule("usatoday_signoff",
             re.compile(
                 r"Thank you for supporting our journalism"
                 r"|Our fact-?check work is supported in part by a grant", _I),
             "section"),
    ],
    "correctiv.org": [
        Rule("correctiv_signoff",
             re.compile(
                 r"Ihre Spende gegen Fake News"
                 r"|CORRECTIV\.?\s*Faktencheck ?ist eine eigenständige "
                 r"Redaktion", _I),
             "section"),
    ],
    "aap.com.au": [
        Rule("aap_signoff",
             re.compile(
                 r"To keep up with our latest fact checks, follow us on"
                 r"|All information, text and images included on the AAP "
                 r"Websites", _I),
             "section"),
    ],
    "vishvasnews.com": [
        # Trailing metadata block opened by "Claim Review :" (space before
        # the colon in the vishvasnews form): claim restatement + "Claimed
        # By :" + "Fact Check : <verdict>" + rating-symbol legend + signoff.
        # section to end-of-doc. Verified trailing corpus-wide (102 docs):
        # the anchor sits at 68-93% of the doc (min fraction 0.68), never
        # mid-doc, and everything after it is non-evidence. Placed first so
        # the whole block (including the "Know the truth!" signoff that
        # vishvas_signoff also matches) goes in one cut.
        Rule("claim_review_block",
             re.compile(r"\bClaim Review\s*:"), "section"),
        Rule("vishvas_signoff",
             re.compile(
                 r"Know the truth!"
                 r"|If you feel any information is doubtful"
                 r"|Contribute to Vishwas News"
                 r"|Help us in our fight against fake news", _I),
             "section"),
        # NOTE: no tipline rule here — vishvasnews's only "Whatsapp tipline"
        # mention is inside the article's Claim-header quote (id=5827);
        # removing it would delete the claim restatement, not meta-text.
    ],
    "newsmobile.in": [
        Rule("newsmobile_signoff",
             re.compile(
                 r"If you want to fact-?check any story, WhatsApp it now"
                 r"|FAKE NEWS BUSTER", _I),
             "section"),
    ],
    "boomlive.in": [
        # Syndication disclaimer, always a trailing parenthetical:
        # "(Except for the headline, this story has not been edited by
        # BOOM staff and is published from a syndicated feed.)"
        Rule("boomlive_syndication",
             re.compile(
                 r"\(?\s*Except for the headline, this story has not been "
                 r"edited by BOOM staff", _I),
             "section"),
        # Ad-blocker nag (some docs are nothing but this nag). token: the
        # three sentences are a fixed block wherever they appear.
        Rule("boomlive_adblock",
             re.compile(
                 r"(?:Our website is made possible by displaying online "
                 r"advertisements to our visitors\.?\s*)?"
                 r"Please consider supporting us by disabling your ad "
                 r"blocker\.?\s*"
                 r"(?:Please reload after ad blocker is disabled\.?)?", _I),
             "token"),
    ],
    "snopes.com": [
        # Meta sign-offs: the ad-submission plea and the "how we work"
        # reading-list intro. sentence mode: both appear mid-doc as well as
        # trailing, so section would eat evidence.
        Rule("snopes_meta_signoff",
             re.compile(
                 r"Feel free to submit ads to us"
                 r"|Happy reading and let us know what else", _I),
             "sentence"),
    ],
    "dw.com": [
        # Trailing translator credit: "This article was adapted from German
        # by <name>" (no terminal period, always at EOD).
        Rule("dw_translator_credit",
             re.compile(
                 r"This article was (?:adapted|translated) from "
                 r"(?:the )?German", _I),
             "section"),
    ],
    "factly.in": [
        # IFCN flash-grant series footer, always the trailing block:
        # "This story is part of a larger series ... This series has been
        # made possible with the flash grant of the IFCN. Read the rest ..."
        Rule("factly_series_footer",
             re.compile(
                 r"This story is part of a larger series"
                 r"|This series has been made possible with the flash grant",
                 _I),
             "section"),
    ],
    "africacheck.org": [
        # Fellowship byline block, trailing: "This report was written by
        # <name>, a ... during the 2023 Africa Check Fellowship ... The
        # fellowship was sponsored by ...". The Fellowship lookahead keeps
        # generic "This report was written by" prose (if any) safe.
        Rule("africacheck_fellowship_byline",
             re.compile(
                 r"This report was written by"
                 r"(?:(?!Africa Check Fellowship)[\s\S]){0,300}?"
                 r"Africa Check Fellowship", _I),
             "section"),
        # Rating-methodology plug: "( Read our overview of how we rate
        # claims here . )"
        Rule("africacheck_rate_overview",
             re.compile(
                 r"\(?\s*Read our overview of how we rate claims here"
                 r"\s*\.?\s*\)?", _I),
             "token"),
    ],
    "leadstories.com": [
        # Trailing related-articles block: "We wrote about <site> before,
        # here are our most recent articles that mention the site:" — the
        # list that follows runs to EOD (median 102 chars, verified
        # trailing corpus-wide).
        Rule("leadstories_wrote_about",
             re.compile(
                 r"We wrote about \S{1,60} before,?\s*"
                 r"here are our most recent articles that mention", _I),
             "section"),
        # Self-promotional related-fact-checks line, e.g. "Other Lead
        # Stories fact checks about Ronaldo can be found here, here and
        # here." Requires the "(An)Other Lead Stories" anchor — a genuine
        # source link ending "can be found here" must survive. token mode
        # (not sentence): these lines are often glued to "(Source: ...)"
        # screenshot captions that PySBD cannot split, so sentence mode
        # would drag in real evidence.
        # Tail is lazy and must end at a period or EOD: a greedy [^.]{0,60}
        # would drag up to 60 chars of following unpunctuated text.
        Rule("leadstories_related",
             re.compile(
                 r"(?:Other|Another)\s+Lead\s+Stories[^.]{0,160}?"
                 r"can\s+be\s+found\s+here[^.]{0,60}?(?:\.|$)", _I),
             "token"),
    ],
    "washingtonpost.com": [
        # Site-UI widget strings scraped into the text: leading/mid
        # "Comment on this story Comment [Gift Article Share]" and the
        # trailing "GiftOutline Gift Article". Pure junk — token mode.
        Rule("wapo_widgets",
             re.compile(
                 r"Comment on this story\s+Comment"
                 r"(?:\s+Gift\s*Article(?:\s+Share)?)?"
                 r"|Gift\s*Outline\s+Gift\s*Article"
                 # The widget is sometimes glued to the next word
                 # ("Gift Article ShareWhat ..."); (?-i:) keeps the
                 # lowercase test case-sensitive under IGNORECASE.
                 r"|Gift\s*Article\s+Share(?!(?-i:[a-z]))"
                 # Social share-bar run. Requires the repeated widget
                 # signature ("Share on Facebook Share on Twitter") so
                 # prose like "share on Facebook and Twitter" never fires.
                 # Leading Shares bounded {0,4}: an unbounded (?:Share\s+)*
                 # is O(n^2) on long "Share Share ..." runs.
                 r"|(?:Share\s+){0,4}Share\s+on\s+Facebook"
                 r"\s+Share\s+on\s+Twitter"
                 r"(?:\s+Share\s+on\s+\w+)*"
                 r"(?:\s+MailSolid\s+Email\s+this\s+link)?", _I),
             "token"),
        # Share-widget metadata dump at EOD: "Share the Facts
        # 2018-02-02 23:09:01 UTC Washington Post -1 -1 -1".
        Rule("wapo_share_the_facts",
             re.compile(
                 r"Share the Facts\s+\d{4}-\d{2}-\d{2}[\s\S]{0,40}?"
                 r"Washington Post(?:\s*-1)*", _I),
             "token"),
        # Fact Checker promo block ("Send us facts to check ... Sign up for
        # The Fact Checker weekly newsletter ... verified signatory ...").
        # token: ~45 docs have real content after the block, so each promo
        # phrase is excised individually wherever it appears.
        Rule("wapo_factchecker_promos",
             re.compile(
                 r"Send us facts to check by filling out this form"
                 r"|Keep tabs on Trump.s promises with our Trump Promise "
                 r"Tracker"
                 r"|Sign up for The Fact Checker weekly newsletter"
                 r"|The Fact Checker is a verified signatory to the "
                 r"International Fact-?Checking Network"
                 r"(?:\s+code of principles)?", _I),
             "token"),
    ],
    "logically.ai": [
        # Stereotyped two-sentence COVID advisory block appended to the
        # COVID subset. token mode: content sometimes follows it.
        Rule("logically_covid_disclaimer",
             re.compile(
                 r"The COVID-19 pandemic has given rise to a lot of\s+"
                 r"potentially dangerous misinformation\.\s*For reliable "
                 r"advice on [^.]*?"
                 r"World Health Organi[sz]ation or your national\s+"
                 r"healthcare authority\.", _I),
             "token"),
    ],
    "factcheckhub.com": [
        # Trailing author-bio block: "<Name> is a fact-checker with (The)
        # FactCheckHub. He has authored several fact checks ... You can
        # reach him via <email>." (6 docs, all identical shape, always
        # runs to end-of-doc). section from the START of the sentence
        # holding the anchor so the author NAME is included in the cut;
        # the `[^.!?]*` pull-back stops at the previous sentence terminator.
        Rule("factcheckhub_bio_block",
             re.compile(
                 r"[^.!?]*fact-?checker with (?:The )?FactCheckHub", _I),
             "section"),
        # Author-bio sign-off sentences ("You can reach him via
        # <email>..."; "<Name> is a Professional Fact-Checker, ...").
        # sentence mode: a preceding bio sentence may survive —
        # precision-first, residual is acceptable.
        Rule("factcheckhub_author_bio",
             re.compile(
                 r"You can reach (?:him|her|them) via"
                 r"|is a Professional Fact-?Checker", _I),
             "sentence"),
        # Trailing syndication credit: "This report was written by Daily
        # Trust, and republished by The FactCheckHub ... See the original
        # piece here."
        Rule("factcheckhub_republished",
             re.compile(
                 r"This report was written by [^.]{0,120}?"
                 r"republished by The FactCheckHub", _I),
             "section"),
    ],
    "checkyourfact.com": [
        Rule("checkyourfact_signoff",
             re.compile(
                 r"Follow [A-Z][a-zA-Z.'\-]+ on Twitter"
                 r"|Have a fact check suggestion"
                 r"|Send ideas to"
                 r"|Send tips to"
                 r"|All content created by the Daily Caller News Foundation"
                 r"|All republished articles must include our logo"
                 r"|For any questions about our guidelines or partnering "
                 r"with us", _I),
             "section"),
    ],
    "newschecker.in": [
        # Same trailing "Claim Review:" metadata block as vishvasnews (1
        # corpus doc, id=6344, anchor at 93% of the doc, to-EOD 377 chars).
        # section to end-of-doc; placed first so it subsumes the signoff.
        Rule("claim_review_block",
             re.compile(r"\bClaim Review\s*:"), "section"),
        Rule("newschecker_signoff",
             re.compile(
                 r"If you would like us to fact[\s-]?check a claim"
                 r"|You can also visit the Contact Us", _I),
             "section"),
        # Mid-article tipline meta-clause: "Newschecker (also) received
        # <this claim/queries/...> on our WhatsApp tipline (+91-9999499044)
        # requesting it to be fact-checked." token with a bounded span:
        # newschecker docs glue sentences through quotes/no-space seams
        # (`churches."We also`, `TiplineFactTo`), so sentence mode drags
        # claim text and real evidence (verified on ids 12, 1264, 3466).
        Rule("newschecker_tipline",
             re.compile(
                 r"(?:Newschecker|We)(?:\s+also)?\s+received\s"
                 r"[^.!?]{0,160}?"
                 r"\(?\+?(?:91[-\s]?)?9999499044\)?"
                 r"(?:,?\s*too)?"
                 r"(?:,?\s*(?:requesting|to)\s[^.!?]{0,50}"
                 r"(?:claim|message|fact[\s-]?checked?))?"
                 r"\.?", _I),
             "token"),
    ],
    "politifact.com": [
        # Trailing "RELATED: <link> RELATED: <link>" run. Each title is
        # capped at 160 chars and may contain internal periods; the run must
        # reach end-of-doc. A mid-article RELATED (followed by >160 chars of
        # evidence) or a top/middle RELATED cluster (followed by the body)
        # cannot form a RELATED-run-to-EOD, so it is left untouched.
        Rule("politifact_related",
             re.compile(r"(?:RELATED:(?:(?!RELATED:)[\s\S]){1,160})+\s*$", _I),
             "section"),
        # Contributor byline sign-off, e.g. "PolitiFact researcher Caryn Baird
        # contributed to this report."
        Rule("politifact_contributor",
             re.compile(
                 r"contributed to this (?:report|fact-?check|article|story)",
                 _I),
             "sentence"),
        # Trailing NC Fact-Checking Project credit block: "This story was
        # produced by the North Carolina Fact-Checking Project, a
        # partnership of ... The NC Local News Lab Fund ..." (verified
        # always trailing, ~310-370 chars to EOD, 26 docs).
        Rule("politifact_nc_project",
             re.compile(
                 r"This story was produced by the North Carolina "
                 r"Fact-?Checking Project", _I),
             "section"),
        # Truth-O-Meter ratings-legend line, e.g. "FALSE — The statement is
        # not accurate. Click here for more on the six PolitiFact ratings..."
        # Case-SENSITIVE: the ALL-CAPS rating word distinguishes it from prose.
        Rule("politifact_ratings_legend",
             re.compile(
                 r"(?:MOSTLY TRUE|MOSTLY FALSE|HALF TRUE|PANTS ON FIRE"
                 r"|TRUE|FALSE)\s*[—–-]\s*The statement[^.]*\."
                 r"(?:\s*Click\s*here\s*for\s*more\s*on\s*the\s*six"
                 r"\s*PolitiFact\s*ratings"
                 r"(?:\s*and\s*how\s*we\s*select\s*facts\s*to\s*check)?\.?)?"),
             "token"),
    ],
}

# domain_of() keys are exact netlocs, so language-edition subdomains need
# their own entries. The Bangla edition shares boomlive.in's boilerplate
# (one corpus doc, an ad-blocker nag).
FOOTER_RULES["bangla.boomlive.in"] = FOOTER_RULES["boomlive.in"]

# ---------------------------------------------------------------------------
# Record drops. Keys are (datasplit, id) of records removed WHOLESALE from
# the cleaned splits: clean_splits drops them BEFORE any stripping and logs
# one {"dropped": true, "reason": ...} diff_log.jsonl entry per key, so drop
# counts stay auditable. Downstream (hybrid pass, make_binary) never sees
# these records. NOTE: ids are positional (assign_ids.py, 0-based within
# each pristine split), so these keys are only valid while the pristine
# files keep their order; clean_splits warns about unmatched keys.

# Badly-scraped thequint.com docs (verified in the pristine corpus
# 2026-07-08): 167/176-char fragments -- an "(Archives of ... claims ...)"
# link parenthetical plus the verdict sentence, no evidence body at all.
_QUINT_SCRAPE_REASON = (
    "badly-scraped thequint fragment: archive-link parenthetical + "
    "verdict sentence only, no evidence body"
)

_UNLOCATABLE_SPAN_REASON = (
    "locator flagged a verdict span that is not locatable verbatim "
    "(LLM quote drift: fused words / emoji / paraphrase) -- residual "
    "verdict text cannot be removed reliably (467303 run, 2026-07-10)"
)

# dw.com multi-sub-claim fact-check compilations reused as "evidence" for
# many different claims: 3-31 sub-claims per doc, each a Claim-quote +
# "DW fact check:"/"Verdict:" label + explanation block. Inherently
# verdict-dense and poor evidence docs; dropping beats further rule-chasing
# (greedy_gate_experiment.md, "Open question" in the v2/v3 results section).
# Verified against the pristine corpus 2026-07-08: dw.com has 24 records =
# 7 unique docs; the 22 keys below are the 6 compilation docs. The 7th
# ("Just how safe is Rwanda...", 5860 chars, test:1997/test:2428) is a
# SINGLE-claim fact check and is kept.
_DW_COMPILATION_REASON = (
    "dw.com multi-sub-claim fact-check compilation (Claim/verdict-label "
    "blocks) reused as evidence for many claims; verdict-dense, poor "
    "evidence doc (greedy_gate_experiment.md §Open question)"
)

DROP_REASONS: dict[tuple[str, int], str] = {
    ("train", 6419): _QUINT_SCRAPE_REASON,  # 167 chars
    ("train", 6703): _QUINT_SCRAPE_REASON,  # 176 chars
    # "Fake news thrives amid Russia-Ukraine war", 19363 chars, 31
    # sub-claims ("DW fact check:" labels) -- 14 identical copies.
    ("test", 188): _DW_COMPILATION_REASON,
    ("train", 1076): _DW_COMPILATION_REASON,
    ("train", 2146): _DW_COMPILATION_REASON,
    ("train", 2514): _DW_COMPILATION_REASON,
    ("train", 3106): _DW_COMPILATION_REASON,
    ("train", 3552): _DW_COMPILATION_REASON,
    ("train", 4105): _DW_COMPILATION_REASON,
    ("train", 5106): _DW_COMPILATION_REASON,
    ("train", 5722): _DW_COMPILATION_REASON,
    ("train", 7311): _DW_COMPILATION_REASON,
    ("train", 7314): _DW_COMPILATION_REASON,
    ("train", 8544): _DW_COMPILATION_REASON,
    ("train", 9641): _DW_COMPILATION_REASON,
    ("val", 1417): _DW_COMPILATION_REASON,
    # "Have the Taliban kept their promises?", 10908 chars, 5 sub-claims.
    ("train", 1623): _DW_COMPILATION_REASON,
    ("val", 29): _DW_COMPILATION_REASON,
    # "Do Vladimir Putin's justifications for going to war ...", 9071
    # chars, 4 sub-claims.
    ("train", 1194): _DW_COMPILATION_REASON,
    ("train", 6425): _DW_COMPILATION_REASON,
    # "Does criminalization prevent abortions?", 8533 chars, 4 sub-claims.
    ("val", 1266): _DW_COMPILATION_REASON,
    # "Crimea bridge blast -- what's real and what's fake?", 8522 chars,
    # 4 sub-claims labelled "Verdict:" (not "DW fact check:"), so it never
    # appeared in hybrid/dw_label_keys.txt; same compilation shape.
    ("train", 7214): _DW_COMPILATION_REASON,
    ("train", 7462): _DW_COMPILATION_REASON,
    # "Is India violating the EU oil embargo?", 5235 chars, 3 sub-claims.
    ("train", 7325): _DW_COMPILATION_REASON,
    # Thin-evidence scrape (user eyeball 2026-07-10): 1145 chars whose
    # English content is claim restatement + FACT CHECK verdict sentence;
    # the remainder is untranslated Hindi portal boilerplate. No usable
    # evidence body after cleaning.
    ("train", 124): (
        "thin-evidence newsmobile scrape: English content is claim "
        "restatement + verdict only; remainder untranslated boilerplate"
    ),
    # Unlocatable-span records (user decision 2026-07-10): in the
    # greedy_v3 production run (job 467303) the locator flagged a verdict
    # span in each of these docs that could not be located verbatim (LLM
    # quote drift: fused words like "wasBarely True", emoji substitution,
    # paraphrase), so the verdict text remains in the doc and no
    # deterministic rule covers it. 9 records corpus-wide -- dropping
    # beats chasing per-doc fused-text repairs. Revisit only if the
    # unlocatable count grows in future runs (see README §issues).
    ("train", 374): _UNLOCATABLE_SPAN_REASON,
    ("train", 1582): _UNLOCATABLE_SPAN_REASON,
    ("train", 1821): _UNLOCATABLE_SPAN_REASON,
    ("train", 3160): _UNLOCATABLE_SPAN_REASON,
    ("train", 4697): _UNLOCATABLE_SPAN_REASON,
    ("train", 4959): _UNLOCATABLE_SPAN_REASON,
    ("train", 6699): _UNLOCATABLE_SPAN_REASON,
    ("train", 8044): _UNLOCATABLE_SPAN_REASON,
    ("val", 2857): _UNLOCATABLE_SPAN_REASON,
}

DROP_KEYS: frozenset[tuple[str, int]] = frozenset(DROP_REASONS)
