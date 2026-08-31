"""Verdict-stripping engine: apply a domain's rules to one doc.

`strip(doc, domain)` returns a StripResult with the cleaned text and the list
of removals applied. A doc whose domain has no rule, or whose rule anchors do
not match, is returned unchanged. All removals are reported for auditability.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    FOOTER_RULES,
    GLOBAL_FOOTER_RULES,
    KEYWORD_DOMAINS,
    KEYWORD_RULES,
    RULES,
    Rule,
)
from src.number_annotation.core.segmentation import segment

# A sentence terminator directly followed by a capital = a missing space in
# the source (e.g. "workers.County"). PySBD can't split there and merges the
# sentences; we sub-split on it so sentence-mode removal stays bounded.
_GLUE_SPLIT = re.compile(r"(?<=[.!?])(?=[A-Z])")


@dataclass
class Removal:
    rule: str
    mode: str
    span: Tuple[int, int]
    removed_text: str
    category: str = "verdict"


@dataclass
class StripResult:
    cleaned: str
    removals: List[Removal] = field(default_factory=list)


def _apply_tagline(doc: str, rule: Rule, removals: List[Removal]) -> str:
    m = rule.anchor.search(doc)
    if not m:
        return doc
    removals.append(
        Removal(rule.name, rule.mode, (m.start(), m.end()), m.group(0))
    )
    return (doc[: m.start()] + doc[m.end():]).rstrip()


def _apply_section(doc: str, rule: Rule, removals: List[Removal]) -> str:
    m = rule.anchor.search(doc)
    if not m:
        return doc
    removals.append(
        Removal(rule.name, rule.mode, (m.start(), len(doc)), doc[m.start():])
    )
    return doc[: m.start()].rstrip()


def _apply_sentence(doc: str, rule: Rule, removals: List[Removal]) -> str:
    matches = list(rule.anchor.finditer(doc))
    if not matches:
        return doc
    sentences = segment(doc).sentences
    # Anchor patterns lead with `[^.]*`, which stops only at `.` and so can
    # span a `!`/`?` boundary that PySBD *does* split on -- dragging the match
    # start back into the previous (evidence) sentence. The trailing `\.`,
    # however, always lands on the verdict sentence's own terminal period, so
    # attribute each hit by the match end (last non-space char), not its start.
    hit_points = []
    for m in matches:
        offset = m.end() - 1
        while offset > m.start() and doc[offset].isspace():
            offset -= 1
        hit_points.append(offset)
    # PySBD sentence spans that contain an anchor hit point.
    hit_spans = [
        (s.start, s.end)
        for s in sentences
        if any(s.start <= p < s.end for p in hit_points)
    ]
    if not hit_spans:
        return doc
    # Within each hit sentence, sub-split on missing-space boundaries and keep
    # only the sub-segment(s) actually holding a hit point. On correctly
    # spaced text _GLUE_SPLIT never matches, so this reduces to the whole
    # sentence (no behaviour change); on glued blobs it bounds the removal.
    remove_spans = []
    for s_start, s_end in hit_spans:
        seg_text = doc[s_start:s_end]
        cuts = (
            [s_start]
            + [s_start + m.start() for m in _GLUE_SPLIT.finditer(seg_text)]
            + [s_end]
        )
        for a, b in zip(cuts, cuts[1:]):
            if any(a <= p < b for p in hit_points):
                remove_spans.append((a, b))
    cleaned = doc
    for start, end in sorted(set(remove_spans), reverse=True):
        removals.append(
            Removal(rule.name, rule.mode, (start, end), doc[start:end])
        )
        cleaned = cleaned[:start] + cleaned[end:]
    # collapse space/tab runs left at the excision seam; preserve newlines
    # so surviving evidence keeps its paragraph structure
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _apply_token(doc: str, rule: Rule, removals: List[Removal]) -> str:
    matches = list(rule.anchor.finditer(doc))
    if not matches:
        return doc
    cleaned = doc
    for m in sorted(matches, key=lambda x: x.start(), reverse=True):
        removals.append(
            Removal(
                rule.name, rule.mode, (m.start(), m.end()), m.group(0)
            )
        )
        cleaned = cleaned[: m.start()] + cleaned[m.end():]
    # repair: collapse space/tab runs the removal created and tidy stray
    # spaces before punctuation; preserve newlines (paragraph structure)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _apply_label_next(doc: str, rule: Rule, removals: List[Removal]) -> str:
    """Remove a `<rating>:` header label AND the sentence following the colon.

    Used for the feedback sites' `DETAILS Misleading: <explanation>` form: the
    rating label leaks the verdict and the sentence after the colon is the
    fact-checker's justification, so both go; later evidence is preserved.
    """
    matches = list(rule.anchor.finditer(doc))
    if not matches:
        return doc
    sentences = segment(doc).sentences
    spans: List[Tuple[int, int]] = []
    for m in matches:
        after = m.end()
        while after < len(doc) and doc[after].isspace():
            after += 1
        end = m.end()
        for s in sentences:
            if s.start <= after < s.end:
                end = s.end
                break
        spans.append((m.start(), max(end, m.end())))
    # merge overlapping spans, then excise right-to-left
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    cleaned = doc
    for start, end in sorted(merged, reverse=True):
        removals.append(
            Removal(rule.name, rule.mode, (start, end), doc[start:end])
        )
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


_DISPATCH = {
    "tagline": _apply_tagline,
    "section": _apply_section,
    "sentence": _apply_sentence,
    "token": _apply_token,
    "label_next": _apply_label_next,
}


def strip(doc: str, domain: str) -> StripResult:
    if not doc or not doc.strip():
        return StripResult(cleaned=doc, removals=[])
    removals: List[Removal] = []
    cleaned = doc

    # 1) Footer rules first (global + domain), tagged category="footer".
    footer_removals: List[Removal] = []
    footer_rules = list(GLOBAL_FOOTER_RULES) + list(
        FOOTER_RULES.get(domain, [])
    )
    for rule in footer_rules:
        handler = _DISPATCH.get(rule.mode)
        if handler is None:
            continue
        cleaned = handler(cleaned, rule, footer_removals)
    for r in footer_removals:
        r.category = "footer"
    removals.extend(footer_removals)

    # 2) Verdict rules, exactly as before (default category="verdict").
    verdict_rules = list(RULES.get(domain, []))
    if domain in KEYWORD_DOMAINS:
        verdict_rules += KEYWORD_RULES
    for rule in verdict_rules:
        handler = _DISPATCH.get(rule.mode)
        if handler is None:
            continue
        cleaned = handler(cleaned, rule, removals)
    return StripResult(cleaned=cleaned, removals=removals)
