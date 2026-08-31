"""Audit-viewer span location (`scripts/footer_audit_view.py`).

Regression for the "not located verbatim" ⚠ artifact: the stripper runs footer
rules BEFORE verdict rules and records each removal's `removed_text` against the
doc state *at its own invocation*, not the pristine doc. When a footer rule
excises text from the MIDDLE of a region that a later verdict `section` rule
then cuts wholesale, the verdict `removed_text` has a gap where the footer text
was, so it is no longer a verbatim substring of the pristine doc. The viewer
must still place both spans (via cumulative replay), not flag a miss.
"""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "footer_audit_view", _REPO / "scripts" / "footer_audit_view.py"
)
fav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fav)


# Mirrors politifact train id 478: a Truth-O-Meter legend (footer) sits inside
# the "Our ruling" block that the `our_ruling` section rule cuts to end.
_EVIDENCE = "Latinos are less likely to call 911. "
_LEGEND = "MOSTLY TRUE- The statement is accurate but needs clarification."
_DOC = (
    _EVIDENCE
    + "Our ruling Harris said studies show 40 percent. "
    + _LEGEND
    + " We rate this claim Mostly True."
)
# Footer token rule: recorded verbatim against the PRISTINE doc (runs first).
# Verdict section rule: recorded against the FOOTER-STRIPPED doc, so its text
# has the legend gap and is NOT a verbatim substring of the pristine doc.
_STRIPPED = _DOC.replace(_LEGEND + " ", "")
_REMOVALS = [
    {"rule": "politifact_ratings_legend", "category": "footer",
     "mode": "token", "removed_text": _LEGEND},
    {"rule": "our_ruling", "category": "verdict", "mode": "section",
     "removed_text": _STRIPPED[_STRIPPED.find("Our ruling"):]},
]


def test_middle_footer_cut_does_not_break_verdict_location():
    spans, misses = fav.locate_spans(_DOC, _REMOVALS)
    assert misses == [], f"expected no misses, got {misses}"


def test_footer_span_covers_the_legend_verbatim():
    spans, _ = fav.locate_spans(_DOC, _REMOVALS)
    footer = [(a, b) for a, b, c in spans if c == "footer"]
    assert len(footer) == 1
    a, b = footer[0]
    assert _DOC[a:b] == _LEGEND


def test_spans_cover_whole_our_ruling_region_and_spare_evidence():
    spans, _ = fav.locate_spans(_DOC, _REMOVALS)
    covered = set()
    for a, b, _cat in spans:
        covered.update(range(a, b))
    region_start = _DOC.find("Our ruling")
    # Every char from "Our ruling" to end is removed (footer or verdict)...
    assert covered == set(range(region_start, len(_DOC)))
    # ...and none of the preceding evidence sentence is touched.
    assert not (covered & set(range(0, region_start)))


# Mirrors factly.in train id 4319: the verdict phrase repeats, and one copy is
# nested inside a larger block that another removal cuts wholesale. Replaying
# left-to-right would greedily consume the nested copy first and corrupt the
# block; the stripper excises right-to-left, so the viewer must too.
_PHRASE = "Hence, the claim is TRUE. "
_BLOCK = "Claim: 267 clusters mandated. " + _PHRASE + "Fact: 267 delivered. "
_DUP_DOC = (
    "Intro evidence. " + _BLOCK + "Body evidence. "
    + _PHRASE + "Middle evidence. " + _PHRASE + "Tail evidence."
)
# Recorded in the stripper's right-to-left excision order (rightmost first).
_DUP_REMOVALS = [
    {"rule": "hence_claim", "category": "verdict", "mode": "sentence",
     "removed_text": _PHRASE},                       # the last standalone copy
    {"rule": "hence_claim", "category": "verdict", "mode": "sentence",
     "removed_text": _PHRASE},                       # the middle standalone copy
    {"rule": "hence_claim", "category": "verdict", "mode": "sentence",
     "removed_text": _BLOCK},                         # block (contains a copy)
]


def test_repeated_phrase_nested_in_block_all_located():
    spans, misses = fav.locate_spans(_DUP_DOC, _DUP_REMOVALS)
    assert misses == [], f"expected no misses, got {misses}"
    covered = set()
    for a, b, _cat in spans:
        covered.update(range(a, b))
    # The block's content is struck (including its nested phrase copy);
    # trailing whitespace isn't struck, which is fine.
    b_start = _DUP_DOC.find(_BLOCK)
    b_end = b_start + len(_BLOCK.rstrip())
    assert set(range(b_start, b_end)) <= covered
    # ...and evidence sentences are spared.
    for ev in ("Intro evidence.", "Body evidence.",
               "Middle evidence.", "Tail evidence."):
        i = _DUP_DOC.find(ev)
        assert not (covered & set(range(i, i + len(ev)))), ev
