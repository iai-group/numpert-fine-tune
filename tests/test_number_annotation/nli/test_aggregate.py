"""Tests for deterministic claim-level aggregation of per-number NLI."""

from src.number_annotation.nli.aggregate import (
    aggregate_record,
    aggregate_records,
)


def _ev(id, relevance=None, relation=None):
    """Evidence-number dict with an optional attached nli result."""
    e = {"id": id}
    if relevance is not None:
        e["nli"] = {"relevance": relevance, "relation": relation,
                    "claim_num": None, "rationale": "r"}
    else:
        e["nli"] = None
    return e


def _rec(label, evs):
    return {"claim_id": "x", "label": label, "evidence_numbers": evs}


def test_supports_only_pair_relation_and_to_false_gate():
    rec = _rec("True", [_ev(0, 5, "supports"), _ev(1, 1, "neutral")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["pair_relation"] == "supports"
    assert agg["relevant_supporting_ids"] == [0]
    assert agg["relevant_refuting_ids"] == []
    assert agg["perturbable"] == {"to_false": True, "to_true": False}


def test_refutes_only_and_to_true_gate_for_false_label():
    rec = _rec("False", [_ev(0, 4, "refutes")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["pair_relation"] == "refutes"
    assert agg["perturbable"] == {"to_false": False, "to_true": True}


def test_both_relations_conflicting():
    rec = _rec("True", [_ev(0, 5, "supports"), _ev(1, 5, "refutes")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["pair_relation"] == "conflicting"


def test_no_relevant_spans_neutral():
    rec = _rec("True", [_ev(0, 2, "supports"), _ev(1, 3, "refutes")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["pair_relation"] == "neutral"
    assert agg["perturbable"] == {"to_false": False, "to_true": False}
    assert agg["max_relevance"] == 3  # over all scored spans


def test_min_relevance_threshold_configurable():
    rec = _rec("True", [_ev(0, 3, "supports")])
    agg = aggregate_record(rec, min_relevance=3)["nli_aggregate"]
    assert agg["pair_relation"] == "supports"
    assert agg["perturbable"]["to_false"] is True


def test_relevant_neutral_spans_counted_but_not_relating():
    rec = _rec("True", [_ev(0, 5, "neutral")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["counts"]["neutral"] == 1
    assert agg["pair_relation"] == "neutral"


def test_errored_spans_counted_never_related():
    rec = _rec("True", [_ev(0), {"id": 1}, _ev(2, 5, "supports")])
    agg = aggregate_record(rec)["nli_aggregate"]
    assert agg["counts"]["errored"] == 2
    assert agg["counts"]["supports"] == 1


def test_conflicting_label_never_perturbable():
    rec = _rec("Conflicting", [_ev(0, 5, "supports"), _ev(1, 5, "refutes")])
    assert aggregate_record(rec)["nli_aggregate"]["perturbable"] == {
        "to_false": False, "to_true": False}


def test_unknown_or_padded_labels():
    padded = _rec("  true ", [_ev(0, 5, "supports")])
    assert aggregate_record(padded)["nli_aggregate"]["perturbable"]["to_false"]
    weird = _rec("half-true", [_ev(0, 5, "supports"), _ev(1, 5, "refutes")])
    agg = aggregate_record(weird)["nli_aggregate"]
    assert agg["pair_relation"] == "conflicting"  # relation is label-free
    assert agg["perturbable"] == {"to_false": False, "to_true": False}


def test_empty_evidence_numbers():
    agg = aggregate_record(_rec("True", []))["nli_aggregate"]
    assert agg["pair_relation"] == "neutral"
    assert agg["max_relevance"] == 0
    assert agg["counts"] == {"supports": 0, "refutes": 0, "neutral": 0,
                             "errored": 0}


def test_aggregate_records_attaches_to_all():
    recs = [_rec("True", [_ev(0, 5, "supports")]), _rec("False", [])]
    out = aggregate_records(recs)
    assert all("nli_aggregate" in r for r in out)
