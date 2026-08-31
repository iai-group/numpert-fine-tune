"""Evaluation: P/R/F1 (exact + overlap), value/type accuracy.

Compares predicted NumberSpans to a curated gold set, and converts
legacy spaCy entities into NumberSpans for the baseline comparison
(spec §5.6).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.number_annotation.core.schema import (
    NumberSpan,
    SPACY_LABEL_TO_VALUE_TYPE,
)


@dataclass
class Scores:
    """Aggregate scores for one document or corpus."""

    exact_precision: float
    exact_recall: float
    exact_f1: float
    overlap_precision: float
    overlap_recall: float
    overlap_f1: float
    value_accuracy: float
    type_accuracy: float


def _f1(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact(a: NumberSpan, b: NumberSpan) -> bool:
    """Return True if two spans share exact offsets."""
    return a.char_start == b.char_start and a.char_end == b.char_end


def _overlaps(a: NumberSpan, b: NumberSpan) -> bool:
    """Return True if two spans overlap at all."""
    return a.char_start < b.char_end and b.char_start < a.char_end


def _match(pred, gold, predicate):
    """Greedily match pred->gold by a predicate; return pairs."""
    used = set()
    pairs = []
    for p in pred:
        for i, g in enumerate(gold):
            if i in used:
                continue
            if predicate(p, g):
                used.add(i)
                pairs.append((p, g))
                break
    return pairs


def _parse_value(text: str) -> Optional[float]:
    """Parse a numeric surface to a float, or None."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def score_spans(
    pred: List[NumberSpan], gold: List[NumberSpan]
) -> Scores:
    """Score predicted spans against gold spans."""
    if not pred and not gold:
        return Scores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    exact_pairs = _match(pred, gold, _exact)
    overlap_pairs = _match(pred, gold, _overlaps)

    def pr(pairs):
        precision = len(pairs) / len(pred) if pred else 0.0
        recall = len(pairs) / len(gold) if gold else 0.0
        return precision, recall

    ex_p, ex_r = pr(exact_pairs)
    ov_p, ov_r = pr(overlap_pairs)

    if overlap_pairs:
        value_acc = sum(
            1 for p, g in overlap_pairs if p.value == g.value
        ) / len(overlap_pairs)
        type_acc = sum(
            1
            for p, g in overlap_pairs
            if p.value_type == g.value_type
        ) / len(overlap_pairs)
    else:
        value_acc = 0.0
        type_acc = 0.0

    return Scores(
        exact_precision=ex_p,
        exact_recall=ex_r,
        exact_f1=_f1(ex_p, ex_r),
        overlap_precision=ov_p,
        overlap_recall=ov_r,
        overlap_f1=_f1(ov_p, ov_r),
        value_accuracy=value_acc,
        type_accuracy=type_acc,
    )


def entities_to_spans(
    entities: List[Dict[str, Any]]
) -> List[NumberSpan]:
    """Convert legacy spaCy entity dicts into NumberSpans."""
    spans = []
    for i, ent in enumerate(entities):
        spans.append(
            NumberSpan(
                id=i,
                surface=ent["text"],
                char_start=ent["start_char"],
                char_end=ent["end_char"],
                value=_parse_value(ent["text"]),
                value_low=None,
                value_high=None,
                value_type=SPACY_LABEL_TO_VALUE_TYPE.get(
                    ent["label"], "other"
                ),
                unit=None,
                modifier="exact",
                sign="+",
                context="",
            )
        )
    return spans
