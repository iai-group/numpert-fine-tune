"""Deterministic structural validation of located spans.

Confirms each span's offsets reproduce its surface verbatim and that no
two spans overlap. This is the deterministic gate behind both the
frontier annotator and (in Phase 2) the fine-tuned annotator.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.number_annotation.core.schema import NumberSpan


@dataclass
class ValidationReport:
    """Outcome of validating a document's spans."""

    ok: bool
    mismatches: List[int] = field(default_factory=list)
    overlaps: List[Tuple[int, int]] = field(default_factory=list)


def validate_spans(
    text: str, spans: List[NumberSpan]
) -> ValidationReport:
    """Validate that spans reproduce their surface and don't overlap."""
    mismatches = [
        s.id
        for s in spans
        if text[s.char_start : s.char_end] != s.surface
    ]

    overlaps: List[Tuple[int, int]] = []
    ordered = sorted(spans, key=lambda s: s.char_start)
    for left, right in zip(ordered, ordered[1:]):
        if left.char_end > right.char_start:
            overlaps.append((left.id, right.id))

    return ValidationReport(
        ok=not mismatches and not overlaps,
        mismatches=mismatches,
        overlaps=overlaps,
    )
