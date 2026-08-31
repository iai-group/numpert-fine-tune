from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    FakeAuditClient,
    audit_records,
    build_audit_records,
    format_audit_report,
    stratified_sample,
)


def test_format_audit_report_summarizes_and_lists_flagged():
    report = {
        "n_audited": 3,
        "flagged": [
            {
                "id": 1,
                "domain": "snopes.com",
                "cleaned_doc": "We rate this claim False.",
                "removed": ["Evidence sentence."],
                "audit": {
                    "still_has_verdict": True,
                    "removed_real_evidence": False,
                },
            },
        ],
    }
    md = format_audit_report(report)
    assert "audited: 3" in md
    assert "flagged: 1" in md
    assert "still_has_verdict (missed leak): 1" in md
    assert "removed_real_evidence (over-removal): 0" in md
    assert "id=1 domain=snopes.com" in md
    assert "We rate this claim False." in md
    assert "Evidence sentence." in md


def test_build_audit_records_joins_cleaned_split_and_difflog():
    cleaned = [
        {"id": 0, "url": "https://www.indiatoday.in/x", "doc": "Evidence."},
        {"id": 1, "url": "https://www.reuters.com/y", "doc": "Untouched."},
    ]
    difflog = [
        {
            "id": 0,
            "domain": "indiatoday.in",
            "removals": [{"removed_text": "JHOOTH ..."}],
        },
    ]
    recs = build_audit_records(cleaned, difflog)
    assert recs[0] == {
        "id": 0,
        "domain": "indiatoday.in",
        "changed": True,
        "cleaned_doc": "Evidence.",
        "removed": ["JHOOTH ..."],
    }
    # untouched record: derives domain from url, no removals, not changed
    assert recs[1]["changed"] is False
    assert recs[1]["domain"] == "reuters.com"
    assert recs[1]["removed"] == []


def test_stratified_sample_covers_each_domain_and_untouched():
    records = (
        [
            {"id": i, "domain": "politifact.com", "changed": True}
            for i in range(20)
        ]
        + [
            {"id": 50 + i, "domain": "snopes.com", "changed": True}
            for i in range(20)
        ]
        + [{"id": 100, "domain": "reuters.com", "changed": False}]
    )
    sample = stratified_sample(records, per_domain=2, untouched=1)
    domains = {r["domain"] for r in sample}
    # both changed domains represented, each capped at per_domain
    assert "politifact.com" in domains
    assert "snopes.com" in domains
    politifact = [r for r in sample if r["domain"] == "politifact.com"]
    snopes = [r for r in sample if r["domain"] == "snopes.com"]
    assert len(politifact) == 2
    assert len(snopes) == 2
    # exactly `untouched` clean docs included
    untouched = [r for r in sample if not r["changed"]]
    assert len(untouched) == 1


def test_audit_isolates_per_record_errors():
    class Boom:
        def check(self, rec):
            if rec["id"] == 2:
                raise RuntimeError("400 Bad Request")
            return {
                "still_has_verdict": False,
                "removed_real_evidence": False,
            }

    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    report = audit_records(records, Boom())
    # the bad record is isolated, the rest still audited
    assert report["n_audited"] == 3
    assert len(report["errored"]) == 1
    assert report["errored"][0]["id"] == 2
    assert "400" in report["errored"][0]["audit_error"]
    assert report["flagged"] == []


def test_audit_flags_missed_leak_and_overremoval():
    client = FakeAuditClient(
        verdicts={
            1: {"still_has_verdict": True, "removed_real_evidence": False},
            2: {"still_has_verdict": False, "removed_real_evidence": True},
            3: {"still_has_verdict": False, "removed_real_evidence": False},
        }
    )
    records = [
        {
            "id": 1,
            "domain": "x",
            "original_doc": "...",
            "cleaned_doc": "...",
            "removed": ["..."],
        },
        {
            "id": 2,
            "domain": "x",
            "original_doc": "...",
            "cleaned_doc": "...",
            "removed": ["..."],
        },
        {
            "id": 3,
            "domain": "x",
            "original_doc": "...",
            "cleaned_doc": "...",
            "removed": ["..."],
        },
    ]
    report = audit_records(records, client)
    flagged_ids = {r["id"] for r in report["flagged"]}
    assert flagged_ids == {1, 2}
    assert report["n_audited"] == 3
