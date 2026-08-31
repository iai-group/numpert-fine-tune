"""spaCy-based numeric locator and best-effort value parsing.

`parse_value` and `ents_to_spans` are pure (no model load) and unit-tested.
`SpacyNumberLocator` lazily loads a configurable spaCy model and returns
`NumberSpan`s with exact char offsets taken from the NER entities.
"""

import re
from typing import List, Optional

from src.number_annotation.core.schema import (
    SPACY_LABEL_TO_VALUE_TYPE,
    NumberSpan,
)

# Numeric NER labels we keep (DATE/TIME included, tagged via value_type).
NUMERIC_LABELS = set(SPACY_LABEL_TO_VALUE_TYPE)

# Magnitude words -> multiplier. Checked largest-first.
_MAGNITUDES = {
    "trillion": 1e12,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_value(surface: str) -> Optional[float]:
    """Best-effort parse of a numeric surface to a float, else None."""
    text = surface.strip().lower()
    if not text:
        return None

    multiplier = 1.0
    for word, factor in _MAGNITUDES.items():
        if word in text:
            multiplier = factor
            text = text.replace(word, " ")
            break

    cleaned = (
        text.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("£", "")
        .replace("€", "")
        .strip()
    )

    match = _NUMBER_RE.search(cleaned)
    if match:
        return float(match.group()) * multiplier

    try:
        from word2number import w2n  # type: ignore[import]

        return float(w2n.word_to_num(cleaned)) * multiplier
    except (ValueError, AttributeError, IndexError):
        return None


def ents_to_spans(ents) -> List[NumberSpan]:
    """Convert numeric spaCy-like entities into NumberSpans.

    `ents` is any iterable of objects exposing `text`, `label_`,
    `start_char`, and `end_char` (e.g. spaCy entity spans). Non-numeric
    labels are skipped. Offsets are copied verbatim; rich fields beyond
    value/value_type take schema defaults (context is filled later).
    """
    spans: List[NumberSpan] = []
    for ent in ents:
        if ent.label_ not in NUMERIC_LABELS:
            continue
        spans.append(
            NumberSpan(
                id=len(spans),
                surface=ent.text,
                char_start=ent.start_char,
                char_end=ent.end_char,
                value=parse_value(ent.text),
                value_low=None,
                value_high=None,
                value_type=SPACY_LABEL_TO_VALUE_TYPE.get(
                    ent.label_, "other"
                ),
                unit=None,
                modifier="exact",
                sign="+",
                context="",
            )
        )
    return spans


class SpacyNumberLocator:
    """Locate numeric mentions in text using a spaCy NER model."""

    def __init__(self, model: str = "en_core_web_trf", use_gpu: bool = False):
        """Store the model name; the model loads on first locate().

        When ``use_gpu`` is set, the GPU is activated before the model loads
        (required for transformer models like ``en_core_web_trf`` to run on
        CUDA). Requires a CUDA-enabled spaCy build (``spacy[cuda12x]``).
        """
        self._model_name = model
        self._use_gpu = use_gpu
        self._nlp = None

    def _ensure_model(self):
        """Load the spaCy model once, lazily."""
        if self._nlp is None:
            import spacy  # type: ignore[import]

            if self._use_gpu:
                spacy.require_gpu()
            self._nlp = spacy.load(self._model_name)
        return self._nlp

    def locate(self, text: str) -> List[NumberSpan]:
        """Return numeric mentions in `text` with exact char offsets."""
        if not text:
            return []
        nlp = self._ensure_model()
        return ents_to_spans(nlp(text).ents)
