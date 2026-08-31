"""Group a record's evidence numbers into one NLI call per sentence window.

Numbers that fall in the same anchor sentence share that sentence's window text,
so they are scored together in one call. Grouping key is always
``tuple(span.block_sentence_ids["sentence"])`` (the anchor sentence); within a
group, numbers are ordered by ``char_start`` and assigned local marker labels
``N0, N1, …``.

The ``window`` argument selects how much surrounding text the model sees:
``"sentence"`` (anchor only), ``"prev_curr"`` (previous + anchor), or
``"window3"`` (previous + anchor + next). The *grouping* and the *marked
numbers* are unchanged across windows — only marked numbers in the anchor
sentence are scored — but a wider window gives the model more context. The
wider windows are ``" ".join(...)`` reconstructions (see ``core/context.py``),
so the anchor sentence begins ``len(context_prev_curr) - len(context_sentence)``
characters into both the ``prev_curr`` and ``window3`` strings; we pass an
adjusted ``sentence_start`` so the marker offsets land correctly.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.number_annotation.core.schema import NumberSpan

WINDOWS = ("sentence", "prev_curr", "window3")

# The LLM emits one result object (with a free-text rationale) per marked
# number, so a dense sentence can overrun the generation budget. Capping the
# markers per call bounds the output size; 8 markers is a few hundred output
# tokens, comfortably under max_tokens=2048.
DEFAULT_MAX_MARKERS_PER_CALL = 8


@dataclass
class SentenceGroup:
    """One NLI call: a sentence window and its marked numbers."""

    sentence_ids: Tuple[int, ...]
    sentence_text: str
    sentence_start: int
    markers: List[Tuple[str, NumberSpan]]


def _has_sentence_context(span: NumberSpan, window: str) -> bool:
    """True if the span carries the window fields we need for ``window``."""
    if (
        span.context_sentence is None
        or span.sentence_start is None
        or not isinstance(span.block_sentence_ids, dict)
        or "sentence" not in span.block_sentence_ids
    ):
        return False
    if window == "prev_curr":
        return span.context_prev_curr is not None
    if window == "window3":
        return (
            span.context_prev_curr is not None
            and span.context_window3 is not None
        )
    return True


def _window_text_and_start(span: NumberSpan, window: str) -> Tuple[str, int]:
    """Return the window text and the start offset to subtract for markers.

    For the anchor-only window the start is the document offset of the anchor
    sentence. For the wider windows the anchor sentence sits ``anchor_offset``
    characters into the reconstructed window string, so we return an *adjusted*
    start (``sentence_start - anchor_offset``); subtracting it from a span's
    document ``char_start`` then yields the span's position within the window.
    """
    if window == "sentence":
        return span.context_sentence, span.sentence_start
    # The prev_curr / window3 strings both begin with "prev + ' ' + anchor",
    # so the anchor's offset within either is len(prev_curr) - len(sentence).
    anchor_offset = len(span.context_prev_curr) - len(span.context_sentence)
    text = (
        span.context_prev_curr
        if window == "prev_curr"
        else span.context_window3
    )
    return text, span.sentence_start - anchor_offset


def group_by_sentence(
    spans: List[NumberSpan],
    window: str = "sentence",
    max_markers_per_call: int = DEFAULT_MAX_MARKERS_PER_CALL,
) -> List[SentenceGroup]:
    """Group ``spans`` by their anchor sentence into SentenceGroups.

    ``window`` selects the context text shown to the model: ``"sentence"``,
    ``"prev_curr"``, or ``"window3"`` (see module docstring).

    A bucket with more than ``max_markers_per_call`` spans is emitted as
    several consecutive groups (in ``char_start`` order) that share the same
    window text/start but carry disjoint marker subsets; labels restart at
    ``N0`` within each chunk (they are call-local). Numbers outside a chunk
    stay visible in the sentence as unmarked context and are scored in their
    own chunk.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    if max_markers_per_call < 1:
        raise ValueError(
            f"max_markers_per_call must be >= 1, got {max_markers_per_call}"
        )

    buckets: Dict[Tuple[int, ...], List[NumberSpan]] = {}
    order: List[Tuple[int, ...]] = []
    for span in spans:
        if not _has_sentence_context(span, window):
            continue
        key = tuple(span.block_sentence_ids["sentence"])
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(span)

    groups: List[SentenceGroup] = []
    for key in order:
        members = sorted(buckets[key], key=lambda s: s.char_start)
        text, start = _window_text_and_start(members[0], window)
        for i in range(0, len(members), max_markers_per_call):
            chunk = members[i:i + max_markers_per_call]
            markers = [(f"N{j}", sp) for j, sp in enumerate(chunk)]
            groups.append(
                SentenceGroup(
                    sentence_ids=key,
                    sentence_text=text,
                    sentence_start=start,
                    markers=markers,
                )
            )
    return groups
