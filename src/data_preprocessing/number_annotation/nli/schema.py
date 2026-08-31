"""Data model + guided-decoding schema for NLI scoring results.

The LLM returns a JSON object with a single ``results`` array. Each item names
the in-sentence marker it refers to (``N0``, ``N1``, …), a 1–5 relevance score,
an NLI relation, an optional claim-number id, and a one-line rationale. The
runner maps ``marker`` back to the real ``NumberSpan.id``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

RELATIONS = {"supports", "refutes", "neutral"}


@dataclass
class NliItem:
    """One scored number as returned by the LLM (marker not yet resolved)."""

    marker: str
    relevance: int
    relation: str
    claim_num: Optional[int]
    rationale: str


def parse_nli_response(data: Dict[str, Any]) -> List[NliItem]:
    """Parse a guided-decoding payload into NliItem objects."""
    return [
        NliItem(
            marker=item["marker"],
            relevance=item["relevance"],
            relation=item["relation"],
            claim_num=item.get("claim_num"),
            rationale=item.get("rationale", ""),
        )
        for item in data.get("results", [])
    ]


NLI_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string"},
                    "relevance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "relation": {
                        "type": "string",
                        "enum": sorted(RELATIONS),
                    },
                    "claim_num": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "marker",
                    "relevance",
                    "relation",
                    "claim_num",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}
