"""Tests for NLI result parsing and the output schema."""

from src.number_annotation.nli.schema import (
    NLI_OUTPUT_SCHEMA,
    NliItem,
    RELATIONS,
    parse_nli_response,
)


def test_relations_enum():
    assert RELATIONS == {"supports", "refutes", "neutral"}


def test_parse_nli_response_maps_items():
    data = {
        "results": [
            {
                "marker": "N0",
                "relevance": 5,
                "relation": "supports",
                "claim_num": 0,
                "rationale": "matches the claim's 50%",
            },
            {
                "marker": "N1",
                "relevance": 1,
                "relation": "neutral",
                "claim_num": None,
                "rationale": "unrelated count",
            },
        ]
    }
    items = parse_nli_response(data)
    assert items == [
        NliItem("N0", 5, "supports", 0, "matches the claim's 50%"),
        NliItem("N1", 1, "neutral", None, "unrelated count"),
    ]


def test_parse_nli_response_empty():
    assert parse_nli_response({"results": []}) == []
    assert parse_nli_response({}) == []


def test_schema_is_array_of_results():
    props = NLI_OUTPUT_SCHEMA["properties"]
    assert props["results"]["type"] == "array"
    item = props["results"]["items"]
    assert set(item["required"]) == {
        "marker",
        "relevance",
        "relation",
        "claim_num",
        "rationale",
    }
    assert item["properties"]["relation"]["enum"] == sorted(RELATIONS)
    assert item["additionalProperties"] is False
