"""Tests for the hybrid-audit HTML report (sampling, join, highlighting)."""

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_audit_report import (  # noqa: E501
    build_hybrid_audit_records,
    diverse_sample,
    highlight_html,
    audit_all,
    render_html,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    FakeAuditClient,
)


def test_build_joins_pristine_hybrid_and_difflogs():
    pristine = [
        {"id": 0, "datasplit": "test", "claim": "C0",
         "doc": "Verdict header. Body. JHOOTH tagline here."},
    ]
    hybrid = [
        {"id": 0, "datasplit": "test", "doc": "Body."},
    ]
    hybrid_difflog = [
        {"id": 0, "datasplit": "test", "domain": "indiatoday.in",
         "kept": ["JHOOTH tagline here."], "rejected": ["Body."],
         "unlocatable": ["paraphrased verdict"], "error": None},
    ]
    rule_difflog = [
        {"id": 0, "datasplit": "test", "domain": "indiatoday.in",
         "removals": [{"removed_text": "Verdict header."}]},
    ]
    recs = build_hybrid_audit_records(
        pristine, hybrid, hybrid_difflog, rule_difflog
    )
    assert len(recs) == 1
    r = recs[0]
    assert r["claim"] == "C0"
    assert r["input_doc"] == "Verdict header. Body. JHOOTH tagline here."
    assert r["cleaned_doc"] == "Body."
    assert r["removed"] == ["JHOOTH tagline here."]      # hybrid (LLM)
    assert r["rule_removed"] == ["Verdict header."]      # rule-based
    assert r["rejected"] == ["Body."]
    assert r["unlocatable"] == ["paraphrased verdict"]
    assert r["domain"] == "indiatoday.in"


def test_build_computes_full_sentence_extents():
    # With the rule-cleaned docs available, each kept quote is replayed
    # through the pipeline's sentence snap so the report can highlight the
    # TRUE removed extent (whole sentence), not just the located fragment.
    pristine = [
        {"id": 3, "datasplit": "test", "claim": "c",
         "doc": "Intro fact stays. We rate this claim False. Tail stays."},
    ]
    rulebased = [
        {"id": 3, "datasplit": "test",
         "doc": "Intro fact stays. We rate this claim False. Tail stays."},
    ]
    hybrid = [
        {"id": 3, "datasplit": "test", "doc": "Intro fact stays. Tail stays."},
    ]
    hybrid_difflog = [
        {"id": 3, "datasplit": "test", "domain": "politifact.com",
         "kept": ["rate this claim False"], "rejected": [],
         "unlocatable": [], "error": None},
    ]
    recs = build_hybrid_audit_records(
        pristine, hybrid, hybrid_difflog, [], rulebased_records=rulebased
    )
    assert recs[0]["removed"] == ["rate this claim False"]  # audit input
    assert recs[0]["removed_extents"] == ["We rate this claim False."]


def test_build_extents_bound_glued_sentences():
    # Glued docs: the extent must be glue-bounded (item 9), so neighbours
    # survive and the extent text matches what the pipeline really cut.
    doc = "Intro fact stays.We rate this claim False.Tail stays."
    pristine = [{"id": 4, "datasplit": "test", "claim": "c", "doc": doc}]
    rulebased = [{"id": 4, "datasplit": "test", "doc": doc}]
    hybrid = [{"id": 4, "datasplit": "test",
               "doc": "Intro fact stays.Tail stays."}]
    hybrid_difflog = [
        {"id": 4, "datasplit": "test", "domain": "politifact.com",
         "kept": ["rate this claim False"], "rejected": [],
         "unlocatable": [], "error": None},
    ]
    recs = build_hybrid_audit_records(
        pristine, hybrid, hybrid_difflog, [], rulebased_records=rulebased
    )
    assert recs[0]["removed_extents"] == ["We rate this claim False."]


def test_build_without_rulebased_records_has_empty_extents():
    pristine = [{"id": 5, "datasplit": "test", "claim": "c", "doc": "A. B."}]
    hybrid = [{"id": 5, "datasplit": "test", "doc": "A."}]
    hybrid_difflog = [
        {"id": 5, "datasplit": "test", "domain": "x",
         "kept": ["B."], "rejected": [], "unlocatable": [], "error": None},
    ]
    recs = build_hybrid_audit_records(pristine, hybrid, hybrid_difflog, [])
    assert recs[0]["removed_extents"] == []


def test_build_skips_entries_with_no_kept():
    pristine = [{"id": 1, "datasplit": "test", "claim": "c", "doc": "d"}]
    hybrid = [{"id": 1, "datasplit": "test", "doc": "d"}]
    hybrid_difflog = [{"id": 1, "datasplit": "test", "domain": "x",
                       "kept": [], "rejected": ["q"], "unlocatable": [],
                       "error": None}]
    assert build_hybrid_audit_records(
        pristine, hybrid, hybrid_difflog, []
    ) == []


def test_diverse_sample_spreads_across_domains():
    recs = (
        [{"domain": "politifact.com", "id": i} for i in range(50)]
        + [{"domain": "africacheck.org", "id": i} for i in range(50)]
        + [{"domain": "indiatoday.in", "id": i} for i in range(3)]
    )
    sample = diverse_sample(recs, total=20)
    assert len(sample) == 20
    domains = {r["domain"] for r in sample}
    # round-robin must include the rare domain, not just the big two
    assert "indiatoday.in" in domains
    assert len(domains) == 3


def test_diverse_sample_caps_at_available():
    recs = [{"domain": "x", "id": 0}, {"domain": "y", "id": 1}]
    assert len(diverse_sample(recs, total=100)) == 2


def test_highlight_wraps_and_escapes():
    doc = 'A <b>bold</b> claim. Verdict incorrect. Tail text.'
    html = highlight_html(doc, ["Verdict incorrect."])
    # the removed phrase is wrapped in the removed span
    assert '<span class="removed">Verdict incorrect.</span>' in html
    # raw angle brackets from the doc are escaped, not rendered
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "<b>bold</b>" not in html


def test_highlight_marks_rejected_in_amber():
    doc = "Keep this. Verdict false. Maybe evidence here. End."
    html = highlight_html(
        doc, removed=["Verdict false."], rejected=["Maybe evidence here."]
    )
    assert '<span class="removed">Verdict false.</span>' in html
    assert '<span class="rejected">Maybe evidence here.</span>' in html


def test_highlight_marks_rule_and_llm_removals_in_distinct_colors():
    doc = "Rule cut here. Body evidence. LLM cut here. Tail."
    html = highlight_html(
        doc, removed=["LLM cut here."], rule_removed=["Rule cut here."]
    )
    assert '<span class="rule">Rule cut here.</span>' in html
    assert '<span class="removed">LLM cut here.</span>' in html


def test_highlight_handles_unlocatable_phrase_gracefully():
    doc = "Plain text only."
    html = highlight_html(doc, ["not present"])
    assert "Plain text only." in html
    assert "removed" not in html


def test_audit_all_attaches_verdict_to_every_record():
    recs = [
        {"id": 1, "domain": "x", "cleaned_doc": "c", "removed": ["r"]},
        {"id": 2, "domain": "x", "cleaned_doc": "c", "removed": ["r"]},
    ]
    client = FakeAuditClient(verdicts={
        1: {"still_has_verdict": True, "removed_real_evidence": False},
    })
    out = audit_all(recs, client)
    assert len(out) == 2
    assert out[0]["audit"]["still_has_verdict"] is True
    # id 2 falls back to the FakeAuditClient default (both false)
    assert out[1]["audit"]["removed_real_evidence"] is False


def test_render_html_highlights_full_extent_over_quote():
    audited = [{
        "id": 8, "datasplit": "test", "domain": "politifact.com",
        "claim": "c",
        "input_doc": "Intro stays. We rate this claim False. Tail stays.",
        "cleaned_doc": "Intro stays. Tail stays.",
        "removed": ["rate this claim False"],
        "removed_extents": ["We rate this claim False."],
        "rule_removed": [], "rejected": [], "unlocatable": [],
        "audit": {"still_has_verdict": False,
                  "removed_real_evidence": False},
    }]
    html = render_html(audited)
    # the FULL sentence extent is wrapped; the bare quote is not separately
    # wrapped (it sits inside the extent span)
    assert '<span class="removed">We rate this claim False.</span>' in html
    assert '<span class="removed">rate this claim False</span>' not in html


def test_render_html_falls_back_to_quote_when_extent_unlocatable():
    # If a rule cut broke the extent's contiguity in the pristine display
    # doc, the located quote is still highlighted.
    audited = [{
        "id": 9, "datasplit": "test", "domain": "politifact.com",
        "claim": "c",
        "input_doc": "Intro stays. We rate this claim False. Tail stays.",
        "cleaned_doc": "Intro stays. Tail stays.",
        "removed": ["rate this claim False"],
        "removed_extents": ["NOT PRESENT IN PRISTINE ANYMORE."],
        "rule_removed": [], "rejected": [], "unlocatable": [],
        "audit": {"still_has_verdict": False,
                  "removed_real_evidence": False},
    }]
    html = render_html(audited)
    assert '<span class="removed">rate this claim False</span>' in html


def test_render_html_includes_booleans_claim_and_doc():
    audited = [{
        "id": 7, "datasplit": "test", "domain": "indiatoday.in",
        "claim": "Some claim",
        "input_doc": "Rule cut. Body here. JHOOTH here.",
        "cleaned_doc": "Body here.", "removed": ["JHOOTH here."],
        "rule_removed": ["Rule cut."],
        "rejected": ["Body here."], "unlocatable": ["ghost quote"],
        "audit": {"still_has_verdict": False, "removed_real_evidence": True},
    }]
    html = render_html(audited)
    assert "Some claim" in html
    assert "still_has_verdict" in html
    assert "removed_real_evidence" in html
    assert '<span class="removed">JHOOTH here.</span>' in html
    assert '<span class="rule">Rule cut.</span>' in html
    assert '<span class="rejected">Body here.</span>' in html
    assert "ghost quote" in html  # unlocatable listed
    assert "indiatoday.in" in html
