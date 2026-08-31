"""Tests for sentence-window context enrichment."""

from src.number_annotation.core.context import enrich
from src.number_annotation.core.schema import NumberSpan
from src.number_annotation.core.segmentation import segment


def _span(char_start, char_end, surface):
    """Build a minimal located NumberSpan for tests."""
    return NumberSpan(
        id=0,
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


# Text with three sentences. Offsets:
#   "One came." -> [0, 9)
#   " We had 5 cats." starts at 9; "5" is at index 16.
#   " Then 6 left." ; "6" near the end.
TEXT = "One came. We had 5 cats. Then 6 left."


def test_anchor_sentence_assigned():
    """A number is tied to the sentence containing its start."""
    five_start = TEXT.index("5")
    spans = enrich([_span(five_start, five_start + 1, "5")], segment(TEXT))
    sp = spans[0]
    assert sp.context_sentence.strip() == "We had 5 cats."
    assert sp.sentence_start <= five_start < sp.sentence_end


def test_prev_curr_block():
    """context_prev_curr joins the previous and anchor sentences."""
    five_start = TEXT.index("5")
    spans = enrich([_span(five_start, five_start + 1, "5")], segment(TEXT))
    assert "One came." in spans[0].context_prev_curr
    assert "We had 5 cats." in spans[0].context_prev_curr


def test_window3_and_block_ids():
    """context_window3 spans prev+anchor+next, with per-window ids."""
    five_start = TEXT.index("5")
    seg = segment(TEXT)
    spans = enrich([_span(five_start, five_start + 1, "5")], seg)
    sp = spans[0]
    assert "One came." in sp.context_window3
    assert "Then 6 left." in sp.context_window3
    # block_sentence_ids is now keyed per context window.
    assert sp.block_sentence_ids == {
        "sentence": [1],
        "prev_curr": [0, 1],
        "window3": [0, 1, 2],
    }


def test_first_sentence_has_no_prev():
    """A number in sentence 0 has prev_curr equal to the anchor only."""
    # Anchor a span at the start sentence (sentence 0 has no preceding one).
    spans = enrich([_span(0, 3, "One")], segment(TEXT))
    sp = spans[0]
    assert sp.context_prev_curr.strip() == "One came."
    assert sp.block_sentence_ids == {
        "sentence": [0],
        "prev_curr": [0],
        "window3": [0, 1],
    }


def test_last_sentence_has_no_next():
    """A number in the final sentence omits the next sentence."""
    six_start = TEXT.index("6")
    seg = segment(TEXT)
    spans = enrich([_span(six_start, six_start + 1, "6")], seg)
    sp = spans[0]
    assert "Then 6 left." in sp.context_window3
    assert sp.block_sentence_ids == {
        "sentence": [2],
        "prev_curr": [1, 2],
        "window3": [1, 2],
    }


def test_two_numbers_same_sentence_share_context():
    """Two numbers in one sentence share sentence_id and windows."""
    text = "We had 5 cats and 7 dogs."
    seg = segment(text)
    five = text.index("5")
    seven = text.index("7")
    spans = enrich(
        [_span(five, five + 1, "5"), _span(seven, seven + 1, "7")], seg
    )
    assert spans[0].sentence_id == spans[1].sentence_id
    assert spans[0].context_sentence == spans[1].context_sentence
    # Identical per-window ids let callers group/dedup the two numbers so a
    # shared window is only evaluated once.
    assert spans[0].block_sentence_ids == spans[1].block_sentence_ids
    assert spans[0].block_sentence_ids["sentence"] == [0]
