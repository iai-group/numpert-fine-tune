from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    FakeAuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.llm_locate import (  # noqa: E501
    FakeLocateClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.locate_eval import (  # noqa: E501
    build_eval_set,
    evaluate,
    format_locate_eval_report,
)


def test_build_eval_set_labels_changed_and_norule():
    cleaned = [
        {"id": 0, "datasplit": "test",
         "url": "https://www.indiatoday.in/x", "doc": "Cleaned A."},
        {"id": 1, "datasplit": "test",
         "url": "https://www.reuters.com/y", "doc": "Untouched B."},
    ]
    difflog = [
        {"id": 0, "datasplit": "test", "domain": "indiatoday.in",
         "removals": [{"removed_text": "JHOOTH ..."}]},
    ]
    evalset = build_eval_set(cleaned, difflog, per_cohort=10)
    by_id = {r["id"]: r for r in evalset}
    assert by_id[0]["cohort"] == "changed"
    assert by_id[0]["cleaned_doc"] == "Cleaned A."
    assert by_id[0]["removed"] == ["JHOOTH ..."]
    assert by_id[1]["cohort"] == "control_norule"
    assert by_id[1]["removed"] == []


def test_build_eval_set_caps_per_cohort():
    cleaned = [
        {"id": i, "datasplit": "test",
         "url": "https://www.politifact.com/z", "doc": f"d{i}"}
        for i in range(20)
    ]
    difflog = [
        {"id": i, "datasplit": "test", "domain": "politifact.com",
         "removals": [{"removed_text": "Our ruling ..."}]}
        for i in range(20)
    ]
    evalset = build_eval_set(cleaned, difflog, per_cohort=5)
    changed = [r for r in evalset if r["cohort"] == "changed"]
    assert len(changed) == 5


def test_evaluate_recall_record_cleared_by_removal():
    eval_records = [
        {"id": 1, "datasplit": "test", "domain": "snopes.com",
         "cohort": "changed",
         "cleaned_doc": "Evidence here. We rate this False.",
         "removed": []},
    ]
    locate = FakeLocateClient(quotes={
        1: {"has_verdict": True, "verdict_quotes": ["We rate this False."]},
    })
    audit = FakeAuditClient(verdicts={
        1: {"still_has_verdict": True, "removed_real_evidence": False},
    })
    report = evaluate(eval_records, locate, audit)
    rec = report["records"][0]
    assert rec["cohort_resolved"] == "recall"
    assert rec["variants"]["phrase"]["located_count"] == 1
    assert report["n_eval"] == 1
    assert "audit_cleared" in report
    assert "false_positive_rate" in report


def test_evaluate_isolates_locate_error():
    class Boom:
        def locate(self, rec):
            raise RuntimeError("boom")

    eval_records = [
        {"id": 9, "datasplit": "test", "domain": "x", "cohort": "changed",
         "cleaned_doc": "doc", "removed": []},
    ]
    audit = FakeAuditClient(verdicts={})
    report = evaluate(eval_records, Boom(), audit)
    assert report["n_eval"] == 1
    assert report["records"][0]["error"]


def test_format_report_has_headline_metrics():
    report = {
        "n_eval": 5, "n_errors": 0,
        "cohorts": {"recall": 3, "clean_control": 2},
        "verbatim_hit_rate": 0.8,
        "audit_cleared": {"phrase": 0.66, "sentence": 1.0},
        "false_positive_rate": 0.0,
        "records": [],
    }
    md = format_locate_eval_report(report)
    assert "verbatim-hit rate" in md
    assert "audit-cleared (phrase)" in md
    assert "audit-cleared (sentence)" in md
    assert "false-positive rate" in md
    assert "recall: 3" in md
