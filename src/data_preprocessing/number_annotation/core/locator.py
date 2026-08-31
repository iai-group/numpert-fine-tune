"""Deterministic resolution of RawNumber surfaces to char offsets.

The annotator returns a surface string plus a short context snippet but
no offsets. This module finds each surface in the document and, when a
surface occurs multiple times, picks the occurrence whose surrounding
window best matches the supplied context — never reusing an offset
already claimed by another number.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from src.number_annotation.core.schema import NumberSpan, RawNumber

_CONTEXT_WINDOW = 40


# Failure reasons (stored on LocationFailure.reason):
#   surface_not_found      — the surface string is absent from the text
#                            entirely (model hallucinated or normalized it,
#                            e.g. emitted "1200" where the text says "1,200")
#   all_occurrences_claimed — the surface DOES occur in the text, but every
#                            occurrence was already claimed by an earlier
#                            number, so there is no free slot left. Usually
#                            means the model emitted the same mention twice.
SURFACE_NOT_FOUND = "surface_not_found"
ALL_OCCURRENCES_CLAIMED = "all_occurrences_claimed"


@dataclass
class LocationFailure:
    """A RawNumber whose surface could not be located."""

    surface: str
    context: str
    reason: str
    # When the failed surface exactly matches an already-located span's
    # surface, the id of that span; otherwise None. A non-None value flags
    # the failure as a duplicate emission of an earlier-located number.
    duplicate_of: Optional[int] = None


def find_occurrences(text: str, surface: str) -> List[int]:
    """Return all start indices of `surface` in `text`."""
    if not surface:
        return []
    starts: List[int] = []
    i = text.find(surface)
    while i != -1:
        starts.append(i)
        i = text.find(surface, i + 1)
    return starts


def _window(text: str, start: int, end: int) -> str:
    """Return the text window around [start, end)."""
    lo = max(0, start - _CONTEXT_WINDOW)
    hi = min(len(text), end + _CONTEXT_WINDOW)
    return text[lo:hi]


def _overlap_score(window: str, context: str) -> int:
    """Return a word-overlap score between window and context."""
    if not context:
        return 0
    win_words = set(window.lower().split())
    ctx_words = set(context.lower().split())
    return len(win_words & ctx_words)


def _is_free(start: int, end: int, used: Set[Tuple[int, int]]) -> bool:
    """Return True if [start, end) does not overlap a used span."""
    for u_start, u_end in used:
        if start < u_end and u_start < end:
            return False
    return True


def _choose(
    text: str,
    surface: str,
    context: str,
    used: Set[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Pick the best free occurrence of surface, or None."""
    candidates = [
        (s, s + len(surface))
        for s in find_occurrences(text, surface)
        if _is_free(s, s + len(surface), used)
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda span: _overlap_score(
            _window(text, span[0], span[1]), context
        ),
    )
    return best


def locate_numbers(
    text: str, raw_numbers: List[RawNumber]
) -> Tuple[List[NumberSpan], List[LocationFailure]]:
    """Locate each RawNumber in `text`, returning spans and failures."""
    spans: List[NumberSpan] = []
    failures: List[LocationFailure] = []
    used: Set[Tuple[int, int]] = set()
    # First located span id per surface, for flagging duplicate failures.
    located_by_surface: dict = {}

    for raw in raw_numbers:
        chosen = _choose(text, raw.surface, raw.context, used)
        if chosen is None:
            # Distinguish "absent from text" from "all slots taken": if the
            # surface occurs at all, every occurrence must already be claimed.
            reason = (
                ALL_OCCURRENCES_CLAIMED
                if find_occurrences(text, raw.surface)
                else SURFACE_NOT_FOUND
            )
            failures.append(
                LocationFailure(
                    surface=raw.surface,
                    context=raw.context,
                    reason=reason,
                    duplicate_of=located_by_surface.get(raw.surface),
                )
            )
            continue
        start, end = chosen
        used.add((start, end))
        span = NumberSpan.from_raw(
            raw, id=len(spans), char_start=start, char_end=end
        )
        located_by_surface.setdefault(raw.surface, span.id)
        spans.append(span)
    return spans, failures
