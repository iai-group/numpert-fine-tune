"""Attach rule-based sentence-window context to located spans.

Given the document's sentence segmentation, each located number is tied to
the sentence containing its start offset, and three window "blocks" are
materialized on it: the anchor sentence (1), previous+anchor (2), and
previous+anchor+next (3). Blocks clamp at document edges. The sentence ids
covered by the 3-window are recorded for future grouping experiments.
"""

from typing import List, Optional

from src.number_annotation.core.schema import NumberSpan
from src.number_annotation.core.segmentation import Segmentation, Sentence


def _anchor_index(
    span: NumberSpan, sentences: List[Sentence]
) -> Optional[int]:
    """Return the index of the sentence containing the span start."""
    for i, sent in enumerate(sentences):
        if sent.start <= span.char_start < sent.end:
            return i
    return None


def enrich(
    spans: List[NumberSpan], seg: Segmentation
) -> List[NumberSpan]:
    """Attach sentence ids, offsets, and window blocks to each span."""
    sents = seg.sentences
    for span in spans:
        idx = _anchor_index(span, sents)
        if idx is None:
            continue
        anchor = sents[idx]
        span.sentence_id = anchor.id
        span.sentence_start = anchor.start
        span.sentence_end = anchor.end

        has_prev = idx - 1 >= 0
        has_next = idx + 1 < len(sents)
        prev = [sents[idx - 1]] if has_prev else []
        nxt = [sents[idx + 1]] if has_next else []

        # Materialize each window's text and the sentence ids it covers, keyed
        # by window name. The per-window ids let callers group/dedup spans by
        # window so a shared window is evaluated only once when several numbers
        # fall in the same sentence(s).
        windows = {
            "sentence": [anchor],
            "prev_curr": prev + [anchor],
            "window3": prev + [anchor] + nxt,
        }
        span.context_sentence = anchor.text
        span.context_prev_curr = " ".join(
            s.text for s in windows["prev_curr"]
        )
        span.context_window3 = " ".join(s.text for s in windows["window3"])
        span.block_sentence_ids = {
            name: [s.id for s in block] for name, block in windows.items()
        }
    return spans
