from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_strip import (  # noqa: E501
    gate_removals,
    hybrid_strip_record,
    hybrid_strip_split,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    FakeLocateClient,
)


class _Audit:
    """Fake audit: flags removed_real_evidence when a removed span is in
    `evidence_spans`; never reports still_has_verdict (unused here)."""

    def __init__(self, evidence_spans):
        self.evidence_spans = set(evidence_spans)

    def check(self, rec):
        removed = rec.get("removed", [])
        bad = any(t in self.evidence_spans for t in removed)
        return {"still_has_verdict": False, "removed_real_evidence": bad}


def test_gate_keeps_clean_verdict_cut():
    doc = "Evidence here. Verdict: False. More evidence."
    res = gate_removals(doc, ["Verdict: False."], _Audit(evidence_spans=[]), 1)
    assert "Verdict: False." not in res.cleaned
    assert "Evidence here." in res.cleaned and "More evidence." in res.cleaned
    assert res.kept == ["Verdict: False."]
    assert res.rejected == []


def test_gate_removes_whole_sentence_for_fragment_quote():
    # locator quotes only a fragment; the gate must excise the WHOLE sentence,
    # leaving no dangling remnant ("Therefore  based on the data.").
    doc = "Keep this. Therefore we rate it unproven based on the data. End."
    res = gate_removals(doc, ["we rate it unproven"], _Audit([]), 9)
    assert res.kept == ["we rate it unproven"]
    assert "we rate it unproven" not in res.cleaned
    assert "based on the data" not in res.cleaned  # whole sentence gone
    assert res.cleaned == "Keep this. End."


def test_gate_rejects_evidence_cut():
    doc = "The wolves attacked the flock. We rate this False."
    res = gate_removals(
        doc,
        ["The wolves attacked the flock."],
        _Audit(evidence_spans=["The wolves attacked the flock."]),
        2,
    )
    assert res.kept == []
    assert res.rejected == ["The wolves attacked the flock."]
    assert res.cleaned == doc  # nothing applied -> unchanged


def test_gate_mixed_keeps_good_reverts_bad():
    doc = "Real evidence sentence. Verdict: False. Tail."
    res = gate_removals(
        doc,
        ["Verdict: False.", "Real evidence sentence."],
        _Audit(evidence_spans=["Real evidence sentence."]),
        3,
    )
    assert res.kept == ["Verdict: False."]
    assert "Real evidence sentence." in res.rejected
    assert "Verdict: False." not in res.cleaned
    assert "Real evidence sentence." in res.cleaned


def test_gate_unlocatable_quote_recorded():
    doc = "Plain evidence only."
    res = gate_removals(doc, ["We rate this False."], _Audit([]), 4)
    assert res.unlocatable == ["We rate this False."]
    assert res.kept == [] and res.cleaned == doc


def test_gate_empty_quotes_no_change():
    doc = "Some evidence."
    res = gate_removals(doc, [], _Audit([]), 5)
    assert res.cleaned == doc
    assert res.kept == [] and res.rejected == [] and res.unlocatable == []


def test_hybrid_strip_record_removes_located_verdict():
    rec = {"id": 1, "datasplit": "test",
           "url": "https://www.snopes.com/x",
           "doc": "Evidence. We rate this False."}
    locate = FakeLocateClient(quotes={
        1: {"has_verdict": True, "verdict_quotes": ["We rate this False."]},
    })
    out = hybrid_strip_record(rec, locate, _Audit(evidence_spans=[]))
    assert out["error"] is None
    assert out["had_locate"] is True
    assert "We rate this False." not in out["doc"]
    assert out["kept"] == ["We rate this False."]
    assert out["domain"] == "snopes.com"


def test_hybrid_strip_record_isolates_errors():
    class Boom:
        def locate(self, rec):
            raise RuntimeError("boom")

    rec = {"id": 2, "datasplit": "test", "url": "https://x/y",
           "doc": "Original rule-cleaned doc."}
    out = hybrid_strip_record(rec, Boom(), _Audit([]))
    assert out["error"]
    assert out["doc"] == "Original rule-cleaned doc."  # fell back, unchanged


def test_hybrid_strip_record_no_verdict_unchanged():
    rec = {"id": 3, "datasplit": "test", "url": "https://x/y",
           "doc": "Just clean evidence."}
    locate = FakeLocateClient(quotes={})  # default has_verdict=False
    out = hybrid_strip_record(rec, locate, _Audit([]))
    assert out["doc"] == "Just clean evidence."
    assert out["kept"] == []


def test_hybrid_strip_split_applies_and_logs():
    records = [
        {"id": 0, "datasplit": "test", "url": "https://www.snopes.com/a",
         "doc": "Evidence. We rate this False."},
        {"id": 1, "datasplit": "test", "url": "https://www.reuters.com/b",
         "doc": "Clean evidence, no verdict."},
    ]
    locate = FakeLocateClient(quotes={
        0: {"has_verdict": True, "verdict_quotes": ["We rate this False."]},
    })
    cleaned, difflog = hybrid_strip_split(
        records, locate, _Audit(evidence_spans=[]), max_workers=2
    )
    assert cleaned[0]["doc"] == "Evidence."
    assert cleaned[1]["doc"] == "Clean evidence, no verdict."  # untouched
    assert len(difflog) == 1
    assert difflog[0]["id"] == 0
    assert difflog[0]["kept"] == ["We rate this False."]
    assert difflog[0]["domain"] == "snopes.com"


def test_hybrid_strip_split_logs_rejected_without_changing_doc():
    records = [
        {"id": 0, "datasplit": "test", "url": "https://www.snopes.com/a",
         "doc": "Evidence sentence. We rate this False."},
    ]
    locate = FakeLocateClient(quotes={
        0: {"has_verdict": True,
            "verdict_quotes": ["Evidence sentence."]},  # the gate will reject
    })
    cleaned, difflog = hybrid_strip_split(
        records, locate, _Audit(evidence_spans=["Evidence sentence."]),
        max_workers=2,
    )
    assert cleaned[0]["doc"] == "Evidence sentence. We rate this False."
    assert difflog[0]["rejected"] == ["Evidence sentence."]
    assert difflog[0]["kept"] == []


def test_chunk_doc_short_doc_single_chunk():
    from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_strip import (  # noqa: E501
        chunk_doc,
    )
    doc = "One sentence. Another sentence."
    assert chunk_doc(doc) == [doc]


def test_chunk_doc_long_doc_verbatim_slices():
    from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_strip import (  # noqa: E501
        chunk_doc,
    )
    doc = " ".join(
        f"Claim number {i} is false because the data disagree."
        for i in range(400)
    )
    chunks = chunk_doc(doc, chunk_chars=2000)
    assert len(chunks) > 1
    for c in chunks:
        assert c in doc  # verbatim substrings (gate requirement)
        assert len(c) <= 2000 + 100  # sentence-aligned, roughly bounded
    # chunks jointly cover the doc's sentence content
    assert sum(len(c) for c in chunks) >= len(doc) - 100


def test_locate_quotes_chunks_and_dedupes():
    from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_strip import (  # noqa: E501
        locate_quotes,
    )

    class RecordingLocate:
        def __init__(self):
            self.calls = []

        def locate(self, rec):
            self.calls.append(rec["cleaned_doc"])
            return {
                "has_verdict": True,
                "verdict_quotes": ["This claim is false.", ""],
            }

    doc = " ".join(
        f"Sub claim {i} was checked. This claim is false."
        for i in range(300)
    )
    client = RecordingLocate()
    has, quotes = locate_quotes(doc, client, rec_id=1)
    assert has is True
    assert quotes == ["This claim is false."]  # deduped, empties dropped
    assert len(client.calls) > 1  # long doc actually chunked
