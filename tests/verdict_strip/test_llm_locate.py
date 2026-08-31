from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    FakeLocateClient,
    LocateClient,
    apply_located_removals,
    locate_span,
)


def test_locate_span_exact():
    doc = "A. Verdict: False. B."
    assert locate_span(doc, "Verdict: False.") == (3, 18)


def test_locate_span_case_insensitive():
    # id=165 shape: locator lowercased the leading word ("the" vs "The")
    doc = "...in the brain.ConclusionThe claim is inaccurate. Symptoms..."
    span = locate_span(doc, "the claim is inaccurate.")
    assert span is not None
    assert doc[span[0]:span[1]] == "The claim is inaccurate."


def test_locate_span_whitespace_insensitive():
    # merged / extra / newline whitespace in the doc vs single-spaced quote
    doc = "Intro. We  rate\nthis claim False. Tail."
    span = locate_span(doc, "We rate this claim False.")
    assert span is not None
    assert doc[span[0]:span[1]] == "We  rate\nthis claim False."


def test_locate_span_returns_none_when_truly_absent():
    assert locate_span("Some unrelated evidence text.", "we rate False") is None


# Real-corpus shape (afp train id=131): the doc pairs a STRAIGHT opening
# quote with a CURLY closing quote; the locator quoted straight quotes on
# both sides -> unlocatable before quote folding.
AFP_131_TAIL = (
    "as it will hack their phone in 10 seconds,” it said in a Facebook "
    "statement on July 18, 2020. \"This message has been identified as "
    "fake.”"
)


def test_locate_span_folds_quote_variants_id131():
    span = locate_span(
        AFP_131_TAIL, '"This message has been identified as fake."'
    )
    assert span is not None
    assert AFP_131_TAIL[span[0]:span[1]] == (
        '"This message has been identified as fake.”'
    )


def test_locate_span_folds_curly_apostrophe():
    doc = "AFP found the video wasn’t filmed in 2020. More text."
    span = locate_span(doc, "the video wasn't filmed in 2020.")
    assert span is not None
    assert doc[span[0]:span[1]] == "the video wasn’t filmed in 2020."


def test_locate_span_strips_one_layer_of_wrapping_quotes():
    # Fallback: the locator wrapped its quote in quotation marks that are
    # not in the doc at all -- strip ONE symmetric layer and retry.
    doc = "Verdict: This message has been identified as fake. Stay alert."
    span = locate_span(
        doc, "“This message has been identified as fake.”"
    )
    assert span is not None
    assert doc[span[0]:span[1]] == (
        "This message has been identified as fake."
    )


def test_locate_span_fallback_straight_quotes():
    doc = "Our ruling stands. The claim is definitely not true. See sources."
    span = locate_span(doc, '"The claim is definitely not true."')
    assert span is not None
    assert doc[span[0]:span[1]] == "The claim is definitely not true."


def test_locate_span_no_strip_when_quotes_not_symmetric():
    # A lone leading quote is part of the needle text, not a wrapper.
    doc = "Message has been identified as fake. Tail."
    assert locate_span(doc, '"Message has been identified as fake') is None


def test_locate_span_quote_folding_no_false_positive():
    # Genuinely absent needle stays not-found through fold AND fallback.
    doc = 'The doc has "no verdict" at all.'
    assert locate_span(doc, '"We rate this False."') is None


def test_locate_span_offsets_stable_with_multichar_lowercase():
    # 'İ'.lower() expands to two chars; the offset map must stay 1:1 so
    # spans after it land on the right original characters.
    doc = "İstanbul report. We rate this claim False."
    span = locate_span(doc, "we rate this claim false.")
    assert span is not None
    assert doc[span[0]:span[1]] == "We rate this claim False."


def test_apply_removes_case_mismatched_quote_not_unlocatable():
    doc = "Body.ConclusionThe claim is inaccurate. More evidence."
    res = apply_located_removals(doc, ["the claim is inaccurate."], "sentence")
    assert res.unlocatable == []                  # was falsely unlocatable
    assert "claim is inaccurate" not in res.cleaned
    assert "More evidence." in res.cleaned


def test_phrase_removes_verbatim_quote_and_repairs_space():
    doc = "Evidence stands. Verdict: False. More evidence here."
    res = apply_located_removals(doc, ["Verdict: False."], "phrase")
    assert "Verdict: False." not in res.cleaned
    assert "Evidence stands." in res.cleaned
    assert "More evidence here." in res.cleaned
    assert "  " not in res.cleaned  # double space collapsed
    assert res.unlocatable == []
    assert len(res.applied) == 1


def test_phrase_skips_unlocatable_quote():
    doc = "Plain evidence with no verdict."
    res = apply_located_removals(doc, ["We rate this False."], "phrase")
    assert res.cleaned == doc          # nothing removed
    assert res.applied == []
    assert res.unlocatable == ["We rate this False."]


def test_phrase_empty_quotes_no_change():
    doc = "Some evidence."
    res = apply_located_removals(doc, [], "phrase")
    assert res.cleaned == doc
    assert res.applied == []
    assert res.unlocatable == []


def test_phrase_multiple_quotes_right_to_left():
    doc = "A. Verdict one. B. Verdict two. C."
    res = apply_located_removals(
        doc, ["Verdict one.", "Verdict two."], "phrase"
    )
    assert "Verdict one." not in res.cleaned
    assert "Verdict two." not in res.cleaned
    assert "A." in res.cleaned and "B." in res.cleaned and "C." in res.cleaned
    assert len(res.applied) == 2


def test_sentence_removes_whole_containing_sentence():
    doc = (
        "The photo spread widely. We rate this claim False. "
        "A reverse search found the origin."
    )
    res = apply_located_removals(doc, ["We rate this claim False"], "sentence")
    assert "We rate this claim False" not in res.cleaned
    assert "photo spread widely" in res.cleaned
    assert "reverse search found the origin" in res.cleaned


# Glued doc (no space after sentence-final punctuation): PySBD merges the
# whole blob into ONE sentence, so an unbounded sentence snap would eat all
# the neighbouring evidence.
GLUED_BLOB = (
    "The dam was built in 1971.Officials confirmed the photo is old."
    "We rate this claim False.A reverse search shows the 2018 flood."
)


def test_sentence_glued_blob_cut_stays_bounded():
    res = apply_located_removals(
        GLUED_BLOB, ["We rate this claim False."], "sentence"
    )
    assert "rate this claim" not in res.cleaned
    assert "built in 1971" in res.cleaned
    assert "photo is old" in res.cleaned
    assert "2018 flood" in res.cleaned


def test_sentence_glued_quote_spanning_boundary_takes_both_subsegments():
    # A located span crossing a glue boundary keeps every sub-segment it
    # overlaps (span-overlap semantics, not a single point).
    res = apply_located_removals(
        GLUED_BLOB, ["photo is old.We rate this claim False."], "sentence"
    )
    assert "photo is old" not in res.cleaned
    assert "rate this claim" not in res.cleaned
    assert "built in 1971" in res.cleaned
    assert "2018 flood" in res.cleaned


# Closing-quote glue (the `churches."We also` shape from the 2026-07-05
# STATUS entry): the terminator is followed by a quote character and THEN
# the capital, so the basic `[.!?][A-Z]` glue pattern misses the boundary.
# PySBD merges this whole doc into one sentence (verified).
QUOTE_GLUED = (
    'Rioters damaged three churches on Aug. 5."We rate this claim False.'
    "A reverse search found the origin."
)


def test_sentence_closing_quote_glue_bounded():
    res = apply_located_removals(
        QUOTE_GLUED, ["We rate this claim False."], "sentence"
    )
    assert "rate this claim" not in res.cleaned
    assert "churches on Aug. 5" in res.cleaned
    assert "reverse search found the origin" in res.cleaned


def test_sentence_initialism_not_split_no_short_fragment():
    # The _GLUE_SPLIT lookbehind must NOT fire inside internal initialisms
    # on well-spaced prose: "U.S." has an uppercase char before each dot, so
    # the acronym stays intact and no stray "S. " fragment is left behind.
    doc = (
        "The U.S. President made the remark in March. "
        "We rate this claim False. A reverse search found the origin."
    )
    res = apply_located_removals(doc, ["We rate this claim False."], "sentence")
    assert "rate this claim" not in res.cleaned
    assert "U.S. President made the remark" in res.cleaned
    assert "reverse search found the origin" in res.cleaned
    # 'U.S.' intact (not split into 'U.' + 'S. President'); one clean cut.
    assert "U.S. President" in res.cleaned
    assert len(res.applied) == 1


def test_sentence_folded_quote_snap_interaction():
    # Quote folding (curly closing quote in doc) THROUGH the sentence-mode
    # snap, not just locate_span: the whole containing sentence is removed.
    doc = (
        "The photo spread widely. "
        "This message has been identified as fake.” "
        "A reverse search found the origin."
    )
    res = apply_located_removals(
        doc, ['"This message has been identified as fake."'], "sentence"
    )
    assert res.unlocatable == []
    assert "identified as fake" not in res.cleaned
    assert "photo spread widely" in res.cleaned
    assert "reverse search found the origin" in res.cleaned


def test_sentence_multiple_folded_quotes_merge():
    # Two folded quotes in one doc: both located via folding, and their
    # sentence snaps apply without eating the evidence between them.
    doc = (
        "Intro evidence here. "
        "The claim is “definitely not true.” "
        "Middle evidence stands. "
        "We rated it “false.” "
        "Closing evidence remains."
    )
    res = apply_located_removals(
        doc,
        ['the claim is "definitely not true."', 'we rated it "false."'],
        "sentence",
    )
    assert res.unlocatable == []
    assert "definitely not true" not in res.cleaned
    assert "rated it" not in res.cleaned
    assert "Intro evidence here" in res.cleaned
    assert "Middle evidence stands" in res.cleaned
    assert "Closing evidence remains" in res.cleaned


def test_sentence_wellspaced_full_snap_unchanged():
    # On correctly spaced text the glue pattern never matches, so the cut
    # is exactly today's full-sentence snap (one applied span).
    doc = (
        "The photo spread widely. We rate this claim False. "
        "A reverse search found the origin."
    )
    res = apply_located_removals(doc, ["rate this claim"], "sentence")
    assert len(res.applied) == 1
    assert res.applied[0][2] == "We rate this claim False. "
    assert res.cleaned == (
        "The photo spread widely. A reverse search found the origin."
    )


def test_sentence_unlocatable_quote_no_change():
    doc = "Just evidence. Nothing to see."
    res = apply_located_removals(doc, ["Verdict: nope"], "sentence")
    assert res.cleaned == doc
    assert res.unlocatable == ["Verdict: nope"]


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_locate_client_parses_quotes():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        content = '{"has_verdict": true, "verdict_quotes": ["We rate False."]}'
        return _FakeResp(
            {"choices": [{"finish_reason": "stop",
                          "message": {"content": content}}]}
        )

    client = LocateClient(model="m", base_url="http://x/v1", post=fake_post)
    out = client.locate({"cleaned_doc": "doc text"})
    assert out["has_verdict"] is True
    assert out["verdict_quotes"] == ["We rate False."]
    assert captured["url"] == "http://x/v1/chat/completions"


def test_locate_client_raises_on_truncation():
    def fake_post(url, json=None, timeout=None):
        return _FakeResp(
            {"choices": [{"finish_reason": "length",
                          "message": {"content": "{"}}]}
        )

    client = LocateClient(model="m", base_url="http://x/v1", post=fake_post)
    try:
        client.locate({"cleaned_doc": "doc"})
        assert False, "expected RuntimeError on truncation"
    except RuntimeError as exc:
        assert "truncated" in str(exc)


def test_fake_locate_client_defaults_empty():
    client = FakeLocateClient(
        quotes={1: {"has_verdict": True, "verdict_quotes": ["x"]}}
    )
    assert client.locate({"id": 1})["verdict_quotes"] == ["x"]
    assert client.locate({"id": 2}) == {
        "has_verdict": False, "verdict_quotes": []
    }
