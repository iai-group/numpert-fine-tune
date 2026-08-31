"""PySBD-based sentence segmentation with global char offsets.

Splits a text into sentences whose `start`/`end` are offsets into the
original text, so `text[start:end] == sentence_text`. PySBD is rule-based
and robust to decimals (e.g. "2. 8 million") and abbreviations, which
surface-punctuation splitting handles poorly.

The sentences always **tile the input exactly**: contiguous, gap-free, and
together covering `[0, len(text))`. PySBD does not always return its pieces
in original order or verbatim (e.g. social-media/hashtag text, decimal/quote
oddities), so we never trust a piece's position blindly — each sentence is a
slice of the *original* text from the previous boundary to the located piece
end, absorbing any skipped region instead of dropping it. Punctuation is
never altered or normalized: every sentence is a verbatim slice of the input.
"""

from dataclasses import dataclass
from typing import List, Sequence

import pysbd  # type: ignore[import]


@dataclass
class Sentence:
    """A sentence with global char offsets into the source text."""

    id: int
    start: int
    end: int
    text: str


@dataclass
class Segmentation:
    """All sentences segmented from a single text."""

    sentences: List[Sentence]


def _tile(text: str, pieces: Sequence[str]) -> List[Sentence]:
    """Reconstruct gap-free, contiguous sentences from pySBD `pieces`.

    Each sentence runs from the previous boundary to the end of the next
    piece located at or after the cursor. Because the sentence text is taken
    as a slice of the *original* `text` (never the piece itself), any region
    skipped by an out-of-order or altered piece is absorbed into the adjacent
    sentence rather than dropped, and any trailing remainder is folded into
    the final sentence. The result always tiles `[0, len(text))` exactly.
    """
    sentences: List[Sentence] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        # Only search forward; a piece that pySBD reordered or altered so it
        # cannot be found ahead of the cursor is skipped (its chars are
        # absorbed by the next located sentence or the trailing remainder).
        idx = text.find(piece, cursor)
        if idx == -1:
            continue
        end = idx + len(piece)
        sentences.append(
            Sentence(
                id=len(sentences),
                start=cursor,
                end=end,
                text=text[cursor:end],
            )
        )
        cursor = end

    if cursor < len(text):
        if sentences:
            last = sentences[-1]
            sentences[-1] = Sentence(
                id=last.id,
                start=last.start,
                end=len(text),
                text=text[last.start:],
            )
        else:
            sentences.append(
                Sentence(id=0, start=0, end=len(text), text=text)
            )
    return sentences


def segment(text: str) -> Segmentation:
    """Segment `text` into sentences that tile it exactly (gap-free)."""
    if not text or not text.strip():
        return Segmentation(sentences=[])

    segmenter = pysbd.Segmenter(language="en", clean=False)
    pieces = segmenter.segment(text)
    return Segmentation(sentences=_tile(text, pieces))
