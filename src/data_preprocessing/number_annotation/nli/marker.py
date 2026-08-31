"""Insert inline [Ni]…[/Ni] markers around numbers in a sentence window.

Offsets on a NumberSpan are document-relative; the sentence window begins at
``sentence_start`` in the document, so a span's position within the window text
is ``char_start - sentence_start``. Markers are emitted left-to-right with a
cursor; any span that overlaps an already-emitted one is skipped.
"""

from typing import List, Tuple

from src.number_annotation.core.schema import NumberSpan


def render_marked_sentence(
    sentence_text: str,
    sentence_start: int,
    markers: List[Tuple[str, NumberSpan]],
) -> str:
    """Return ``sentence_text`` with each (label, span) wrapped in markers."""
    ordered = sorted(markers, key=lambda m: m[1].char_start)
    parts: List[str] = []
    cursor = 0
    for label, span in ordered:
        start = span.char_start - sentence_start
        end = span.char_end - sentence_start
        if start < cursor:
            # Overlaps a span already emitted; skip to keep output well-formed.
            continue
        parts.append(sentence_text[cursor:start])
        parts.append(f"[{label}]{sentence_text[start:end]}[/{label}]")
        cursor = end
    parts.append(sentence_text[cursor:])
    return "".join(parts)
