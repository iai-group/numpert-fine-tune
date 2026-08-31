"""Smoke tests for the NLI HTML report."""

from src.number_annotation.nli.report import generate_nli_html


def _record():
    text = "Intro. reached 40% up from 1.2 million people."
    return {
        "claim": "More than 50 percent used welfare.",
        "claim_id": "test_000001",
        "doc": text,
        "claim_numbers": [],
        "evidence_numbers": [
            {"id": 0, "surface": "40%", "char_start": 15, "char_end": 18,
             "value_type": "percent",
             "nli": {"relevance": 5, "relation": "supports", "claim_num": 0,
                     "rationale": "matches 50%"}},
            {"id": 1, "surface": "1.2 million", "char_start": 27,
             "char_end": 38, "value_type": "count", "nli": None},
        ],
    }


def test_generate_html_contains_claim_and_scores():
    html = generate_nli_html([_record()])
    assert "<html" in html.lower()
    assert "More than 50 percent used welfare." in html
    assert "test_000001" in html
    # The marked number renders with its relevance/relation in the title.
    assert "supports" in html
    assert "rel=5" in html


def test_generate_html_handles_unscored_number():
    html = generate_nli_html([_record()])
    # The unscored number (nli=None) still appears as plain doc text.
    assert "1.2 million" in html


def test_generate_html_empty_records():
    html = generate_nli_html([])
    assert "<html" in html.lower()


def test_summary_block_counts():
    records = [{
        "claim": "c", "claim_id": "id1", "label": "True", "doc": "9 9",
        "evidence_numbers": [
            {"id": 0, "char_start": 0, "char_end": 1,
             "nli": {"relevance": 5, "relation": "supports",
                     "claim_num": None, "rationale": "r"}},
            {"id": 1, "char_start": 2, "char_end": 3, "nli": None,
             "nli_error": "boom"},
        ],
        "nli_aggregate": {
            "pair_relation": "supports",
            "counts": {"supports": 1, "refutes": 0, "neutral": 0,
                       "errored": 1},
            "max_relevance": 5,
            "relevant_supporting_ids": [0], "relevant_refuting_ids": [],
            "perturbable": {"to_false": True, "to_true": False},
        },
    }]
    html = generate_nli_html(records)
    assert "Summary" in html
    assert "1/2" in html            # attached / total spans
    assert "supports: 1" in html    # relation distribution
    assert "to_false: 1" in html    # perturbable rollup


def test_html_max_records_caps_blocks_but_not_summary():
    def rec(i):
        return {
            "claim": "c", "claim_id": f"id{i}", "label": "True", "doc": "9",
            "evidence_numbers": [
                {"id": 0, "char_start": 0, "char_end": 1,
                 "nli": {"relevance": 5, "relation": "supports",
                         "claim_num": None, "rationale": "r"}}],
        }
    records = [rec(i) for i in range(10)]
    html = generate_nli_html(records, max_records=3, seed=0)
    assert html.count("<section style='margin:1.5em") == 3  # record blocks
    assert "Showing 3 of 10 records" in html
    assert "10/10" in html  # summary still spans all records
    # cap off -> all records rendered, no note
    full = generate_nli_html(records)
    assert full.count("<section style='margin:1.5em") == 10
    assert "Showing" not in full
