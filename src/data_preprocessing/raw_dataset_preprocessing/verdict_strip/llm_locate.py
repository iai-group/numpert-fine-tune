"""LLM-assisted verdict locate (spike).

The model only LOCATES verdict text (returns verbatim quotes); this module
removes a quote ONLY when it is a verbatim substring of the doc, so every edit
is auditable and bounded to text proven to exist. Two removal strategies:
  - phrase  : excise the exact quoted span.
  - sentence: excise the PySBD sentence(s) the span overlaps, bounded on
    glued/malformed docs by sub-splitting on missing-space boundaries
    (see _glue_bounded_spans).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import requests

from src.number_annotation.core.segmentation import segment


@dataclass
class RemovalResult:
    cleaned: str
    applied: List[Tuple[int, int, str]] = field(default_factory=list)
    unlocatable: List[str] = field(default_factory=list)


def _repair_whitespace(text: str) -> str:
    """Collapse space/tab runs and tidy space-before-punctuation; keep
    newlines."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    return text.strip()


# Scraped docs and the locator's quotes freely mix straight and typographic
# quote characters (afp train id=131: the doc has `fake.”`, the locator
# quoted `fake."`). Fold every variant to its ASCII form on BOTH needle and
# haystack before matching. Every key is a single code point replaced 1:1,
# so the normalized->original offset map stays position-preserving.
_QUOTE_FOLD = {
    "“": '"',  # “
    "”": '"',  # ”
    "„": '"',  # „
    "«": '"',  # «
    "»": '"',  # »
    "‘": "'",  # ‘
    "’": "'",  # ’
    "‚": "'",  # ‚
    "‹": "'",  # ‹
    "›": "'",  # ›
}


def _norm_with_map(text: str) -> Tuple[str, List[int]]:
    """Lowercase, fold quote variants, drop whitespace; return (norm, map)
    where map[i] is the original index of the i-th normalized char."""
    chars: List[str] = []
    idx_map: List[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        folded = _QUOTE_FOLD.get(ch, ch).lower()
        # .lower() can expand to 2 chars (e.g. 'İ' -> 'i̇'), which would
        # desync the 1:1 offset map; keep such chars as-is instead.
        chars.append(folded if len(folded) == 1 else ch)
        idx_map.append(i)
    return "".join(chars), idx_map


def _locate_folded(doc: str, quote: str):
    """One locate attempt: exact match, then normalized (case/whitespace/
    quote-variant insensitive) match. Spans index the ORIGINAL `doc`."""
    if not quote:
        return None
    idx = doc.find(quote)
    if idx != -1:
        return (idx, idx + len(quote))
    ndoc, nmap = _norm_with_map(doc)
    nq, _ = _norm_with_map(quote)
    if not nq:
        return None
    j = ndoc.find(nq)
    if j == -1:
        return None
    return (nmap[j], nmap[j + len(nq) - 1] + 1)


def locate_span(doc: str, quote: str):
    """Find `quote` in `doc`, tolerating case/whitespace/quote differences.

    Returns an (start, end) span into the ORIGINAL `doc`, or None. Tries an
    exact match first (fast path), then falls back to a case-insensitive,
    whitespace-insensitive, quote-variant-insensitive match — handling
    scraped text where the locator's quote differs from the doc only by
    capitalisation (e.g. "the claim" vs "The claim"), merged/extra
    whitespace ("ConclusionThe", double spaces, newlines), or typographic
    quotes (`fake.”` vs `fake."`). As a last resort, if the needle is
    wrapped in one symmetric layer of quotation marks the doc lacks, that
    layer is stripped and the match retried.
    """
    if not quote:
        return None
    span = _locate_folded(doc, quote)
    if span is not None:
        return span
    if len(quote) >= 2:
        first = _QUOTE_FOLD.get(quote[0], quote[0])
        last = _QUOTE_FOLD.get(quote[-1], quote[-1])
        if first == last and first in ('"', "'"):
            return _locate_folded(doc, quote[1:-1])
    return None


# A sentence terminator glued straight onto the next sentence's capital
# (optionally with a closing quote in between: `churches."We also`) marks a
# missing space PySBD cannot split on -- it merges the sentences, and an
# unbounded sentence snap would drag neighbouring evidence into the cut.
#
# The lookbehind REQUIRES a lowercase letter or digit before the terminator.
# Without it the pattern also fires INSIDE internal initialisms on properly
# spaced prose ("Washington D.C. and", "U.S. President") -- the dot after an
# uppercase letter is not a sentence boundary, and splitting there leaves a
# stray "D." / "S. " fragment and manufactures a bogus seam (corpus replay of
# current kept quotes: 67 such <30-char dangling fragments). Every intended
# glue boundary ends on lowercase/digit (`False.A`, `old.We`, `churches."We`,
# `1971.Officials`, `Aug. 5."We`, `workers.County`), so the guard preserves
# them while dropping the acronym false-splits.
#
# This INTENTIONALLY diverges further from stripper._GLUE_SPLIT, which keeps
# the bare `(?<=[.!?])` (and lacks the quote tolerance too). The stripper's
# handler is domain-anchored and narrow-surface, so its own splits have not
# shown this failure; aligning it to this pattern is a possible future
# follow-up but is out of scope here (would change rule-pass output
# corpus-wide).
_GLUE_SPLIT = re.compile(r"(?<=[a-z0-9][.!?])(?=[\"'“”’]?[A-Z])")


def _glue_bounded_spans(
    doc: str, snap_start: int, snap_end: int, start: int, end: int
) -> List[Tuple[int, int]]:
    """Bound a sentence-snapped removal on glued/malformed docs.

    Sub-split the snapped region [snap_start, snap_end) on `_GLUE_SPLIT`
    and keep only the sub-segment(s) overlapping the located span
    [start, end). On correctly spaced text the guarded pattern never
    matches (a space follows the terminator, and its lowercase/digit
    lookbehind also skips internal initialisms like "U.S."), so this
    returns the whole snap -- no behaviour change; on glued blobs it
    bounds the cut.
    """
    seg_text = doc[snap_start:snap_end]
    cuts = (
        [snap_start]
        + [snap_start + m.start() for m in _GLUE_SPLIT.finditer(seg_text)]
        + [snap_end]
    )
    kept = [
        (a, b)
        for a, b in zip(cuts, cuts[1:])
        if a < end and start < b
    ]
    # The located span always overlaps the snap, so `kept` is non-empty;
    # fall back to the full snap rather than ever emit an empty cut.
    return kept or [(snap_start, snap_end)]


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent (start, end) spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def apply_located_removals(
    doc: str, quotes: List[str], strategy: str
) -> RemovalResult:
    """Remove each verbatim-locatable quote from `doc` per `strategy`."""
    sentences = segment(doc).sentences if strategy == "sentence" else None
    spans: List[Tuple[int, int]] = []
    unlocatable: List[str] = []
    for q in quotes:
        if not q:
            continue
        found = locate_span(doc, q)
        if found is None:
            unlocatable.append(q)
            continue
        start, end = found
        if strategy == "sentence":
            hit = [
                s for s in sentences
                if s.start < end and start < s.end
            ]
            if hit:
                spans.extend(
                    _glue_bounded_spans(
                        doc, hit[0].start, hit[-1].end, start, end
                    )
                )
            else:
                spans.append((start, end))
        else:
            spans.append((start, end))
    merged = _merge_spans(spans)
    applied: List[Tuple[int, int, str]] = []
    cleaned = doc
    for start, end in sorted(merged, reverse=True):
        applied.append((start, end, doc[start:end]))
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = _repair_whitespace(cleaned)
    return RemovalResult(
        cleaned=cleaned,
        applied=list(reversed(applied)),
        unlocatable=unlocatable,
    )


LOCATE_SYSTEM_PROMPT = (
    "You extract fact-checker VERDICT language from evidence text. Quote "
    "VERBATIM (character-for-character, exactly as it appears) every phrase "
    "or sentence that states the fact-checker's own rating/ruling on the "
    "claim - e.g. an explicit rating (True/False/Misleading/Mixture/Pants on "
    "Fire), 'we rate ...', 'Verdict <rating>', '<N> Pinocchios', or a "
    "standalone 'VERDICT' header line. Do NOT quote the claim itself, the "
    "evidence, the analysis/reasoning, or sources. If there is no explicit "
    "verdict, return has_verdict=false and an empty list."
)

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_verdict": {"type": "boolean"},
        "verdict_quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["has_verdict", "verdict_quotes"],
    "additionalProperties": False,
}


def _locate_user_prompt(rec: Dict[str, Any]) -> str:
    return (
        f"EVIDENCE TEXT:\n{rec.get('cleaned_doc', '')}\n\n"
        "Return JSON: {\"has_verdict\": bool, \"verdict_quotes\": [exact "
        "substrings]}"
    )


@dataclass
class LocateClient:
    model: str
    base_url: str
    # 300 s read-timed-out on dw.com's ~19k-char compilations under load
    # (9 records in the 2026-07-07 run).
    timeout: int = 600
    # 512 truncated the response on dw.com's ~19k-char multi-fact-check
    # compilations (13 finish_reason=length errors in the 2026-07-03 run).
    max_tokens: int = 2048
    post: Callable[..., Any] = requests.post

    def locate(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        content = f"{LOCATE_SYSTEM_PROMPT}\n\n{_locate_user_prompt(rec)}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "locate",
                    "strict": True,
                    "schema": LOCATE_SCHEMA,
                },
            },
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = self.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "vLLM locate response truncated at max_tokens="
                f"{self.max_tokens} (finish_reason=length). Increase "
                "max_tokens."
            )
        return json.loads(choice["message"]["content"])


@dataclass
class FakeLocateClient:
    quotes: Dict[Any, Dict[str, Any]]

    def locate(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        return self.quotes.get(
            rec.get("id"), {"has_verdict": False, "verdict_quotes": []}
        )
