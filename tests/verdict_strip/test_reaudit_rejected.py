import json

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    AUDIT_SYSTEM_PROMPT,
    GREEDY_AUDIT_SYSTEM_PROMPT,
    AuditClient,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.reaudit_rejected import (  # noqa: E501
    load_rejected_cuts,
    reaudit_all,
    summarize,
)


class _StubClient:
    """Audit double returning a fixed removed_real_evidence verdict."""

    def __init__(self, rejects: bool):
        self.rejects = rejects
        self.seen = []

    def check(self, rec):
        self.seen.append(rec)
        return {
            "still_has_verdict": False,
            "removed_real_evidence": self.rejects,
        }


def test_load_rejected_cuts(tmp_path):
    log = tmp_path / "llm_diff_log.jsonl"
    log.write_text(
        json.dumps(
            {
                "id": 7,
                "datasplit": "test",
                "domain": "politifact.com",
                "kept": ["We rate this False."],
                "rejected": ["The claim is accurate but needs context."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cuts = load_rejected_cuts(str(log))
    assert cuts == [
        {
            "id": 7,
            "datasplit": "test",
            "domain": "politifact.com",
            "quote": "The claim is accurate but needs context.",
        }
    ]


def test_reaudit_flip_counted():
    doc = (
        "The senator said crime rose 40 percent. Police data show a 4 "
        "percent rise. The claim is accurate but needs context."
    )
    cuts = [
        {
            "id": 1,
            "datasplit": "test",
            "domain": "politifact.com",
            "quote": "The claim is accurate but needs context.",
        }
    ]
    docs = {("test", 1): doc}
    baseline = _StubClient(rejects=True)
    candidate = _StubClient(rejects=False)
    results = reaudit_all(cuts, docs, baseline, candidate, max_workers=1)
    assert results[0]["status"] == "ok"
    # the excised doc handed to the audit no longer contains the span
    assert "needs context" not in baseline.seen[0]["cleaned_doc"]
    assert "Police data" in baseline.seen[0]["cleaned_doc"]
    s = summarize(results)
    assert s["baseline_rejects_again"] == 1
    assert s["candidate_rejects"] == 0
    assert s["flipped_to_accept"] == 1


def test_reaudit_unlocatable_and_missing_doc():
    cuts = [
        {"id": 1, "datasplit": "t", "domain": "d", "quote": "not present"},
        {"id": 2, "datasplit": "t", "domain": "d", "quote": "whatever"},
    ]
    docs = {("t", 1): "Some completely different text."}
    results = reaudit_all(
        cuts, docs, _StubClient(True), _StubClient(False), max_workers=1
    )
    assert results[0]["status"] == "unlocatable"
    assert results[1]["status"] == "missing_doc"
    s = summarize(results)
    assert s["ok"] == 0 and s["unlocatable"] == 1 and s["missing_doc"] == 1


def test_audit_client_uses_custom_system_prompt():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"still_has_verdict": false, '
                                '"removed_real_evidence": false}'
                            },
                        }
                    ]
                }

        return R()

    client = AuditClient(
        model="m",
        base_url="http://x/v1",
        system_prompt=GREEDY_AUDIT_SYSTEM_PROMPT,
        post=fake_post,
    )
    client.check({"id": 1, "cleaned_doc": "text", "removed": ["span"]})
    content = captured["payload"]["messages"][0]["content"]
    assert content.startswith(GREEDY_AUDIT_SYSTEM_PROMPT)
    assert AUDIT_SYSTEM_PROMPT.splitlines()[-1] not in content
