"""End-to-end orchestration test with a fake client."""

import json
import re

from src.number_annotation.nli.client import FakeNliClient
from src.number_annotation.nli.run_nli import (
    sample_records,
    score_records,
    write_meta,
)
from src.number_annotation.nli.schema import NliItem


def _record():
    text = "reached 40% up from 1.2 million people."
    return {
        "claim": "More than 50 percent used welfare.",
        "claim_id": "test_000001",
        "doc": "Intro. " + text,
        "claim_numbers": [
            {"id": 0, "surface": "More than 50 percent", "char_start": 0,
             "char_end": 20, "value": 50.0, "value_low": None,
             "value_high": None, "value_type": "percent", "unit": None,
             "modifier": "exact", "sign": "+", "context": ""},
        ],
        "evidence_numbers": [
            {"id": 0, "surface": "40%", "char_start": 15, "char_end": 18,
             "value": None, "value_low": None, "value_high": None,
             "value_type": "percent", "unit": None, "modifier": "exact",
             "sign": "+", "context": "", "sentence_id": 1,
             "sentence_start": 7, "sentence_end": 7 + len(text),
             "context_sentence": text, "context_prev_curr": text,
             "context_window3": text, "block_sentence_ids": {"sentence": [1]}},
            {"id": 1, "surface": "1.2 million", "char_start": 27,
             "char_end": 38, "value": None, "value_low": None,
             "value_high": None, "value_type": "count", "unit": None,
             "modifier": "exact", "sign": "+", "context": "",
             "sentence_id": 1, "sentence_start": 7,
             "sentence_end": 7 + len(text), "context_sentence": text,
             "context_prev_curr": text, "context_window3": text,
             "block_sentence_ids": {"sentence": [1]}},
        ],
    }


def test_score_records_attaches_nli_by_marker():
    # Both numbers live in one sentence -> one call, two markers.
    client = FakeNliClient(default=[
        NliItem("N0", 5, "supports", 0, "matches 50%"),
        NliItem("N1", 1, "neutral", None, "unrelated count"),
    ])
    out = score_records([_record()], client, max_workers=2)
    ev = out[0]["evidence_numbers"]
    assert ev[0]["nli"] == {
        "relevance": 5, "relation": "supports", "claim_num": 0,
        "rationale": "matches 50%",
    }
    assert ev[1]["nli"]["relation"] == "neutral"


def test_score_records_handles_unmatched_markers_gracefully():
    client = FakeNliClient(default=[NliItem("N0", 4, "supports", 0, "x")])
    out = score_records([_record()], client, max_workers=1)
    ev = out[0]["evidence_numbers"]
    assert ev[0]["nli"]["relevance"] == 4
    assert ev[1]["nli"] is None  # no marker N1 returned


class _RaisingClient:
    """Client whose every call fails, to exercise per-group isolation."""

    def score(self, claim, claim_numbers, marked):
        raise RuntimeError("truncated at max_tokens")


def test_score_records_isolates_failing_group():
    # A failing sentence-group must not abort the run: its spans get
    # nli=None plus an nli_error breadcrumb, and scoring still completes.
    out = score_records([_record()], _RaisingClient(), max_workers=1)
    ev = out[0]["evidence_numbers"]
    assert ev[0]["nli"] is None
    assert ev[1]["nli"] is None
    assert "truncated" in ev[0]["nli_error"]
    assert "truncated" in ev[1]["nli_error"]


def test_score_records_succeeds_despite_one_bad_record():
    # One failing record among several leaves the others fully scored.
    good = FakeNliClient(default=[NliItem("N0", 5, "supports", 0, "ok")])

    class _MixedClient:
        def __init__(self):
            self._calls = 0

        def score(self, claim, claim_numbers, marked):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("boom")
            return good.score(claim, claim_numbers, marked)

    out = score_records(
        [_record(), _record()], _MixedClient(), max_workers=1
    )
    errored = [
        e for r in out for e in r["evidence_numbers"] if e.get("nli_error")
    ]
    attached = [
        e for r in out for e in r["evidence_numbers"] if e.get("nli")
    ]
    assert errored  # the bad group surfaced an error
    assert attached  # the good record still scored


def test_sample_records_is_deterministic():
    records = [{"claim_id": f"c{i}"} for i in range(10)]
    a = sample_records(records, n=3, seed=0)
    b = sample_records(records, n=3, seed=0)
    assert a == b
    assert len(a) == 3


def test_sample_records_n_zero_returns_all():
    records = [{"claim_id": f"c{i}"} for i in range(5)]
    assert sample_records(records, n=0, seed=0) == records


def test_score_records_writes_roundtrips(tmp_path):
    client = FakeNliClient(default=[NliItem("N0", 5, "supports", 0, "ok")])
    out = score_records([_record()], client, max_workers=1)
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(out))
    assert json.loads(path.read_text())[0]["evidence_numbers"][0]["nli"]

# --- chunking + aggregation + meta sidecar --------------------------------


class _EchoClient:
    """Scores every distinct marker present in the marked sentence."""

    def score(self, claim, claim_numbers, marked):
        markers = dict.fromkeys(re.findall(r"\[(N\d+)\]", marked))
        return [NliItem(m, 5, "supports", None, "echo") for m in markers]


def _dense_record(n=10):
    text = "9 " * n
    return {
        "claim": "c", "claim_id": "dense", "label": "True", "doc": text,
        "claim_numbers": [],
        "evidence_numbers": [
            {"id": i, "surface": "9", "char_start": 2 * i,
             "char_end": 2 * i + 1, "value": None, "value_low": None,
             "value_high": None, "value_type": "count", "unit": None,
             "modifier": "exact", "sign": "+", "context": "",
             "sentence_id": 0, "sentence_start": 0,
             "sentence_end": len(text), "context_sentence": text,
             "block_sentence_ids": {"sentence": [0]}}
            for i in range(n)
        ],
    }


def test_dense_record_chunked_zero_errors():
    out = score_records([_dense_record(10)], _EchoClient(),
                        max_workers=1, max_markers=4)
    ev = out[0]["evidence_numbers"]
    assert all(e["nli"] for e in ev)  # every span scored, none dropped
    assert not any(e.get("nli_error") for e in ev)


def test_score_records_aggregates_by_default():
    out = score_records([_dense_record(3)], _EchoClient(), max_workers=1)
    agg = out[0]["nli_aggregate"]
    assert agg["pair_relation"] == "supports"
    assert agg["perturbable"]["to_false"] is True


def test_score_records_aggregation_can_be_disabled():
    out = score_records([_dense_record(3)], _EchoClient(), max_workers=1,
                        aggregate=False)
    assert "nli_aggregate" not in out[0]


def test_write_meta_sidecar(tmp_path):
    output = tmp_path / "scored.json"
    meta_path = write_meta(
        str(output), model="m", window="sentence", max_markers=8,
        min_relevance=4, n_records=1, n_spans=3, n_attached=3, n_errored=0,
        wall_seconds=1.25, input_path="in.json",
    )
    assert meta_path == str(output) + ".meta.json"
    meta = json.loads((tmp_path / "scored.json.meta.json").read_text())
    for key in ("slurm_job_id", "model", "window", "max_markers",
                "min_relevance", "n_records", "n_spans", "n_attached",
                "n_errored", "wall_seconds", "input", "output"):
        assert key in meta
    assert meta["output"] == str(output)
    assert meta["wall_seconds"] == 1.2
