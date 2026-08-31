"""Tests for grouping evidence numbers into per-sentence NLI calls."""

import pytest

from src.number_annotation.core.schema import NumberSpan
from src.number_annotation.nli.grouping import SentenceGroup, group_by_sentence
from src.number_annotation.nli.marker import render_marked_sentence


def _span(id, char_start, char_end, surface, sentence_id, sentence_start,
          context_sentence, block_ids, context_prev_curr=None,
          context_window3=None, block_dict=None):
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
        sentence_id=sentence_id,
        sentence_start=sentence_start,
        sentence_end=sentence_start + len(context_sentence),
        context_sentence=context_sentence,
        context_prev_curr=context_prev_curr,
        context_window3=context_window3,
        block_sentence_ids=block_dict or {"sentence": block_ids},
    )


def test_two_numbers_one_sentence_single_group():
    text = "reached 40% up from 1.2 million people."
    a = _span(0, 8, 11, "40%", 3, 0, text, [3])
    b = _span(1, 20, 31, "1.2 million", 3, 0, text, [3])
    groups = group_by_sentence([a, b])
    assert len(groups) == 1
    g = groups[0]
    assert isinstance(g, SentenceGroup)
    assert g.sentence_ids == (3,)
    assert g.sentence_text == text
    assert g.sentence_start == 0
    assert [label for label, _ in g.markers] == ["N0", "N1"]
    assert [sp.id for _, sp in g.markers] == [0, 1]


def test_numbers_in_different_sentences_split():
    a = _span(0, 8, 11, "40%", 3, 0, "reached 40% today.", [3])
    b = _span(1, 5, 8, "1.2", 7, 100, "and 1.2 next.", [7])
    groups = group_by_sentence([a, b])
    assert len(groups) == 2
    assert {g.sentence_ids for g in groups} == {(3,), (7,)}


def test_markers_ordered_by_char_start():
    text = "from 1.2 million to 40% over time."
    later = _span(0, 20, 23, "40%", 3, 0, text, [3])
    earlier = _span(1, 5, 16, "1.2 million", 3, 0, text, [3])
    groups = group_by_sentence([later, earlier])
    g = groups[0]
    # N0 is assigned to the earliest by char_start regardless of input order.
    assert g.markers[0][0] == "N0"
    assert g.markers[0][1].id == 1
    assert g.markers[1][1].id == 0


def test_spans_without_sentence_context_skipped():
    bare = NumberSpan(
        id=0, surface="5", char_start=0, char_end=1, value=5.0,
        value_low=None, value_high=None, value_type="count", unit=None,
        modifier="exact", sign="+", context="",
    )  # no sentence fields
    assert group_by_sentence([bare]) == []


# --- wider-window (prev_curr / window3) support --------------------------

# A document whose sentences tile gap-free; context windows are the
# " ".join(...) reconstructions core/context.py produces (an extra space is
# inserted at each sentence boundary).
PREV = "Intro text here. "          # doc[0:17]
ANCHOR = "Sales hit 40% now. "      # doc[17:36], "40%" at doc[27:30]
NEXT = "The end."                   # doc[36:44]
PREV_CURR = PREV + " " + ANCHOR     # join inserts one space
WINDOW3 = PREV + " " + ANCHOR + " " + NEXT


def _windowed_span():
    return _span(
        0, 27, 30, "40%", 1, 17, ANCHOR, [1],
        context_prev_curr=PREV_CURR,
        context_window3=WINDOW3,
        block_dict={"sentence": [1], "prev_curr": [0, 1], "window3": [0, 1, 2]},
    )


def test_window3_text_and_marker_offsets_align():
    g = group_by_sentence([_windowed_span()], window="window3")[0]
    assert g.sentence_text == WINDOW3
    # The adjusted start, when subtracted from the doc char_start, must locate
    # "40%" inside the reconstructed window string.
    marked = render_marked_sentence(
        g.sentence_text, g.sentence_start, g.markers
    )
    assert "[N0]40%[/N0]" in marked
    assert marked == WINDOW3.replace("40%", "[N0]40%[/N0]")


def test_prev_curr_text_and_marker_offsets_align():
    g = group_by_sentence([_windowed_span()], window="prev_curr")[0]
    assert g.sentence_text == PREV_CURR
    marked = render_marked_sentence(
        g.sentence_text, g.sentence_start, g.markers
    )
    assert marked == PREV_CURR.replace("40%", "[N0]40%[/N0]")


def test_sentence_window_is_default_and_unchanged():
    g = group_by_sentence([_windowed_span()])[0]
    assert g.sentence_text == ANCHOR
    assert g.sentence_start == 17
    marked = render_marked_sentence(
        g.sentence_text, g.sentence_start, g.markers
    )
    assert marked == ANCHOR.replace("40%", "[N0]40%[/N0]")


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        group_by_sentence([_windowed_span()], window="paragraph")


# --- dense-group chunking (max_markers_per_call) --------------------------


def _dense_bucket(n, sentence_id=3, sentence_start=0, id_base=0):
    """n one-char spans sharing one anchor sentence ("9 9 9 ...")."""
    text = "9 " * n
    return [
        _span(id_base + i, 2 * i, 2 * i + 1, "9", sentence_id,
              sentence_start, text, [sentence_id])
        for i in range(n)
    ]


def test_dense_bucket_chunks_into_capped_groups():
    groups = group_by_sentence(_dense_bucket(42), max_markers_per_call=8)
    assert len(groups) == 6  # ceil(42/8)
    assert [len(g.markers) for g in groups] == [8, 8, 8, 8, 8, 2]
    # Union of chunk markers == original bucket, char_start order preserved.
    all_ids = [sp.id for g in groups for _, sp in g.markers]
    assert all_ids == list(range(42))
    # Labels restart at N0 within each chunk (labels are call-local).
    for g in groups:
        assert [label for label, _ in g.markers] == [
            f"N{i}" for i in range(len(g.markers))
        ]
    # All chunks share the same sentence text/start/ids.
    assert {g.sentence_text for g in groups} == {groups[0].sentence_text}
    assert {g.sentence_start for g in groups} == {0}
    assert {g.sentence_ids for g in groups} == {(3,)}


def test_bucket_at_or_below_cap_single_group():
    for n in (8, 3):
        groups = group_by_sentence(_dense_bucket(n), max_markers_per_call=8)
        assert len(groups) == 1
        assert len(groups[0].markers) == n


def test_chunking_applies_per_bucket_not_globally():
    a = _dense_bucket(3, sentence_id=3)
    b = _dense_bucket(4, sentence_id=7, sentence_start=50, id_base=100)
    groups = group_by_sentence(a + b, max_markers_per_call=3)
    sizes = [(g.sentence_ids, len(g.markers)) for g in groups]
    assert sizes == [((3,), 3), ((7,), 3), ((7,), 1)]


def test_chunks_share_window_text_for_window3():
    import dataclasses

    first = _windowed_span()
    # A second number in the same anchor sentence.
    second = dataclasses.replace(first, id=1, char_start=31, char_end=34)
    groups = group_by_sentence(
        [first, second], window="window3", max_markers_per_call=1
    )
    assert len(groups) == 2
    assert all(g.sentence_text == WINDOW3 for g in groups)
    assert len({g.sentence_start for g in groups}) == 1


def test_invalid_max_markers_rejected():
    with pytest.raises(ValueError):
        group_by_sentence(_dense_bucket(2), max_markers_per_call=0)


def test_window3_skipped_when_window_fields_missing():
    # A span that only has the anchor-sentence fields cannot serve window3.
    bare_anchor = _span(0, 8, 11, "40%", 3, 0, "reached 40% today.", [3])
    assert group_by_sentence([bare_anchor], window="window3") == []
    # ...but is still grouped under the default sentence window.
    assert len(group_by_sentence([bare_anchor])) == 1
