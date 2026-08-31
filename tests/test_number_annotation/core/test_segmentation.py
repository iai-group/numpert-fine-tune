"""Tests for PySBD sentence segmentation with global offsets."""

import pytest

from src.number_annotation.core.segmentation import _tile, segment


class TestSegment:
    """Tests for segment."""

    def test_offsets_reproduce_text(self):
        """Each sentence's offsets slice its exact text."""
        text = "The tower is tall. It is in Paris."
        seg = segment(text)
        assert len(seg.sentences) == 2
        for sent in seg.sentences:
            assert text[sent.start:sent.end] == sent.text

    def test_ids_are_contiguous(self):
        """Sentence ids start at 0 and increment by 1."""
        text = "One. Two. Three."
        seg = segment(text)
        assert [s.id for s in seg.sentences] == [0, 1, 2]

    @pytest.mark.xfail(
        reason=(
            "Known PySBD limitation: a misplaced-whitespace decimal like "
            "'2. 8 million' is split at the period. The entity-overlap "
            "guard that would re-join it is deferred per the spec; the "
            "global-offset logic itself is correct (see "
            "test_offsets_reproduce_text)."
        ),
        strict=False,
    )
    def test_decimal_with_misplaced_space_stays_one_sentence(self):
        """'2. 8 million' does not split into two sentences."""
        text = "Revenue grew by 2. 8 million dollars last year."
        seg = segment(text)
        assert len(seg.sentences) == 1

    def test_abbreviation_not_split(self):
        """An abbreviation like 'Dr.' does not end a sentence."""
        text = "Dr. Smith arrived. He was late."
        seg = segment(text)
        assert len(seg.sentences) == 2
        assert "Dr. Smith arrived." in seg.sentences[0].text

    def test_empty_text_yields_no_sentences(self):
        """Empty or whitespace-only text yields no sentences."""
        assert segment("").sentences == []
        assert segment("   \n  ").sentences == []


def _assert_tiles(text, seg):
    """Sentences tile the whole input exactly: contiguous, gap-free, exact."""
    sents = seg.sentences
    assert sents, "non-empty text must yield at least one sentence"
    assert sents[0].start == 0
    assert sents[-1].end == len(text)
    for a, b in zip(sents, sents[1:]):
        assert a.end == b.start  # contiguous: no gaps, no overlaps
    for s in sents:
        assert text[s.start:s.end] == s.text  # exact slice of the original
    assert "".join(s.text for s in sents) == text  # nothing dropped


class TestTilingInvariant:
    """segment() must cover the whole input with no dropped regions."""

    def test_segment_tiles_input_exactly(self):
        """A battery of texts each tile their input exactly."""
        for text in [
            "The tower is tall. It is in Paris.",
            "One. Two. Three.",
            "Dr. Smith arrived. He was late.",
            "Revenue grew by 2. 8 million dollars last year.",
            "  Leading space. Trailing too.   ",
            "A “quote with $4. 99. ” And more text after.",
            "No terminal punctuation",
        ]:
            _assert_tiles(text, segment(text))


class TestTileReconstruction:
    """_tile covers the input even when pySBD pieces are reordered/altered."""

    def test_out_of_order_pieces_leave_no_gap(self):
        """A piece emitted out of original order does not skip a region."""
        text = "Alpha one. Beta two. Gamma three."
        pieces = ["Alpha one.", "Gamma three.", "Beta two."]  # reordered
        sents = _tile(text, pieces)
        assert sents[0].start == 0 and sents[-1].end == len(text)
        for a, b in zip(sents, sents[1:]):
            assert a.end == b.start
        assert "".join(s.text for s in sents) == text  # nothing dropped

    def test_unfound_piece_is_skipped_not_dropped(self):
        """A piece absent from the text is skipped; its region is absorbed."""
        text = "Alpha one. Beta two."
        pieces = ["Alpha one.", "NOT IN TEXT", "Beta two."]
        sents = _tile(text, pieces)
        assert "".join(s.text for s in sents) == text
        assert sents[-1].end == len(text)

    def test_trailing_remainder_absorbed(self):
        """Text after the last located piece is absorbed, not dropped."""
        text = "Alpha one. Beta two. tail"
        pieces = ["Alpha one.", "Beta two."]  # nothing for ' tail'
        sents = _tile(text, pieces)
        assert sents[-1].end == len(text)
        assert sents[-1].text.endswith("tail")
