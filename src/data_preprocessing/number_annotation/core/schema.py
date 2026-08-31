"""Data model for annotated numeric mentions.

`RawNumber` is what the annotator LLM returns (no character offsets).
`NumberSpan` adds deterministically-resolved offsets and a stable id.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

VALUE_TYPES = {
    "count",
    "percent",
    "money",
    "ordinal",
    "date_year",
    "duration",
    "fraction",
    "ratio",
    "other",
}

MODIFIERS = {"exact", "approx", "at_least", "at_most", "range"}

SIGNS = {"+", "-"}

# Mapping back to the legacy spaCy NER labels, for baseline comparison.
SPACY_LABEL_TO_VALUE_TYPE = {
    "CARDINAL": "count",
    "QUANTITY": "count",
    "MONEY": "money",
    "PERCENT": "percent",
    "ORDINAL": "ordinal",
    "DATE": "date_year",
    "TIME": "duration",
}


@dataclass
class RawNumber:
    """A numeric mention as produced by the annotator (no offsets)."""

    surface: str
    value: Optional[float]
    value_low: Optional[float]
    value_high: Optional[float]
    value_type: str
    unit: Optional[str]
    modifier: str
    sign: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass
class NumberSpan:
    """A located numeric mention with verified character offsets."""

    id: int
    surface: str
    char_start: int
    char_end: int
    value: Optional[float]
    value_low: Optional[float]
    value_high: Optional[float]
    value_type: str
    unit: Optional[str]
    modifier: str
    sign: str
    context: str
    sentence_id: Optional[int] = None
    sentence_start: Optional[int] = None
    sentence_end: Optional[int] = None
    context_sentence: Optional[str] = None
    context_prev_curr: Optional[str] = None
    context_window3: Optional[str] = None
    # Sentence ids covered by each context window, keyed by window name
    # ("sentence", "prev_curr", "window3"). Lets callers group/dedup spans by
    # window so a shared window is evaluated only once when several numbers
    # fall in the same sentence(s).
    block_sentence_ids: Optional[Dict[str, List[int]]] = None

    @classmethod
    def from_raw(
        cls, raw: RawNumber, id: int, char_start: int, char_end: int
    ) -> "NumberSpan":
        """Build a NumberSpan from a RawNumber and resolved offsets."""
        return cls(
            id=id,
            surface=raw.surface,
            char_start=char_start,
            char_end=char_end,
            value=raw.value,
            value_low=raw.value_low,
            value_high=raw.value_high,
            value_type=raw.value_type,
            unit=raw.unit,
            modifier=raw.modifier,
            sign=raw.sign,
            context=raw.context,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


def raw_from_dict(data: Dict[str, Any]) -> RawNumber:
    """Deserialize a RawNumber from a dict."""
    return RawNumber(
        surface=data["surface"],
        value=data.get("value"),
        value_low=data.get("value_low"),
        value_high=data.get("value_high"),
        value_type=data["value_type"],
        unit=data.get("unit"),
        modifier=data["modifier"],
        sign=data["sign"],
        context=data.get("context", ""),
    )


def span_from_dict(data: Dict[str, Any]) -> NumberSpan:
    """Deserialize a NumberSpan from a dict."""
    return NumberSpan(
        id=data["id"],
        surface=data["surface"],
        char_start=data["char_start"],
        char_end=data["char_end"],
        value=data.get("value"),
        value_low=data.get("value_low"),
        value_high=data.get("value_high"),
        value_type=data["value_type"],
        unit=data.get("unit"),
        modifier=data["modifier"],
        sign=data["sign"],
        context=data.get("context", ""),
        sentence_id=data.get("sentence_id"),
        sentence_start=data.get("sentence_start"),
        sentence_end=data.get("sentence_end"),
        context_sentence=data.get("context_sentence"),
        context_prev_curr=data.get("context_prev_curr"),
        context_window3=data.get("context_window3"),
        block_sentence_ids=data.get("block_sentence_ids"),
    )
