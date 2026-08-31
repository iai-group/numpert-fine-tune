"""Tests for inline [Ni] marker insertion."""

from src.number_annotation.core.schema import NumberSpan
from src.number_annotation.nli.marker import render_marked_sentence


def _span(id, char_start, char_end, surface):
    return NumberSpan(
        id=id,
        surface=surface,
        char_start=char_start,
        char_end=char_end,
        value=None,
        value_low=None,
        value_high=None,
        value_type="count",
        unit=None,
        modifier="exact",
        sign="+",
        context="",
    )


def test_single_number():
    text = "About 40% of households were reached."
    # "40%" sits at chars 6-9 within the sentence (sentence_start=0).
    sp = _span(0, 6, 9, "40%")
    out = render_marked_sentence(text, 0, [("N0", sp)])
    assert out == "About [N0]40%[/N0] of households were reached."


def test_two_numbers_doc_relative_offsets():
    # Sentence starts at doc offset 100; offsets are doc-relative.
    text = "reached 40% up from 1.2 million people."
    sp_a = _span(0, 108, 111, "40%")        # 100 + 8 .. 100 + 11
    sp_b = _span(1, 120, 131, "1.2 million")  # 100 + 20 .. 100 + 31
    out = render_marked_sentence(text, 100, [("N0", sp_a), ("N1", sp_b)])
    assert out == "reached [N0]40%[/N0] up from [N1]1.2 million[/N1] people."


def test_leading_quote_offset():
    # Mirrors claim_numbers where the sentence opens with a quote at offset 0.
    text = '"More than 50 percent of immigrants ..."'
    sp = _span(0, 1, 21, "More than 50 percent")
    out = render_marked_sentence(text, 0, [("N0", sp)])
    assert out == '"[N0]More than 50 percent[/N0] of immigrants ..."'


def test_overlapping_span_skipped():
    text = "between 2 and 3 percent"
    sp_a = _span(0, 0, 23, "between 2 and 3 percent")
    sp_b = _span(1, 8, 9, "2")  # nested inside sp_a -> skipped
    out = render_marked_sentence(text, 0, [("N0", sp_a), ("N1", sp_b)])
    assert out == "[N0]between 2 and 3 percent[/N0]"


def test_no_markers_returns_text():
    assert render_marked_sentence("plain text", 0, []) == "plain text"
