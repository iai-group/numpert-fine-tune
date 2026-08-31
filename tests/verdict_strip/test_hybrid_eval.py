from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    FakeAuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    FakeLocateClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_eval import (  # noqa: E501
    aggregate_by_label,
    build_label_balanced_set,
    evaluate_hybrid,
)


def _cleaned(n_per):
    recs = []
    rid = 0
    for lab in ("True", "False", "Conflicting"):
        for _ in range(n_per):
            recs.append({
                "id": rid, "datasplit": "test", "label": lab,
                "url": "https://www.politifact.com/x", "doc": f"doc {rid}",
            })
            rid += 1
    return recs


def test_build_label_balanced_set_balances_and_caps():
    cleaned = _cleaned(50)  # 50 each
    difflog = [
        {"id": 0, "datasplit": "test", "domain": "politifact.com",
         "removals": [{"removed_text": "Our ruling ..."}]},
    ]
    evalset = build_label_balanced_set(cleaned, difflog, per_label=10)
    from collections import Counter
    by_label = Counter(r["label"] for r in evalset)
    assert by_label == {"True": 10, "False": 10, "Conflicting": 10}
    # record 0 was in the diff log -> cohort changed with removed spans
    rec0 = [r for r in evalset if r["id"] == 0][0]
    assert rec0["cohort"] == "changed"
    assert rec0["removed"] == ["Our ruling ..."]


def test_aggregate_by_label_computes_improvement():
    # 2 records: one rule-residual that the LLM clears, one already clean
    records = [
        {"id": 1, "label": "False", "error": None,
         "initial_still_has_verdict": True,
         "variants": {"phrase": {"located_count": 1,
                                 "post_still_has_verdict": False,
                                 "post_removed_real_evidence": False}}},
        {"id": 2, "label": "False", "error": None,
         "initial_still_has_verdict": False,
         "variants": {"phrase": {"located_count": 0,
                                 "post_still_has_verdict": False,
                                 "post_removed_real_evidence": False}}},
    ]
    agg = aggregate_by_label(records)
    o = agg["overall"]
    assert o["n"] == 2
    assert o["residual_after_rules"] == 0.5   # 1 of 2 flagged after rules
    assert o["residual_after_hybrid"] == 0.0  # LLM cleared it
    assert o["improvement"] == 0.5
    assert o["locate_recall_on_residual"] == 1.0  # the 1 residual was located
    assert "False" in agg["by_label"]


def test_evaluate_hybrid_runs_concurrently_with_fakes():
    eval_records = [
        {"id": 1, "datasplit": "test", "label": "False", "domain": "snopes.com",
         "cohort": "changed",
         "cleaned_doc": "Evidence. We rate this False.", "removed": []},
        {"id": 2, "datasplit": "test", "label": "True", "domain": "x",
         "cohort": "control_norule", "cleaned_doc": "Clean evidence.",
         "removed": []},
    ]
    locate = FakeLocateClient(quotes={
        1: {"has_verdict": True, "verdict_quotes": ["We rate this False."]},
    })
    audit = FakeAuditClient(verdicts={
        1: {"still_has_verdict": True, "removed_real_evidence": False},
    })
    report = evaluate_hybrid(eval_records, locate, audit, max_workers=2)
    assert report["overall"]["n"] == 2
    assert report["overall"]["n_errors"] == 0
