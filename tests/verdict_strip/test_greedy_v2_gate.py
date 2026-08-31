"""Item-8 greedy-v2 gate prompt + shared PROMPTS wiring.

The v2 prompt leans further toward acceptance (removes fact-bearing spans
whose framing resolves the claim); still_has_verdict is unchanged from v1.
hybrid_strip (--gate-prompt) and reaudit_rejected (--baseline/--candidate-
prompt) both resolve their choice through the shared PROMPTS dict.
"""
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.audit_llm import (  # noqa: E501
    AUDIT_SYSTEM_PROMPT,
    GREEDY_AUDIT_SYSTEM_PROMPT,
    GREEDY_V2_AUDIT_SYSTEM_PROMPT,
    GREEDY_V3_AUDIT_SYSTEM_PROMPT,
    PROMPTS,
    AuditClient,
)


def test_prompts_dict_names():
    assert set(PROMPTS) == {"baseline", "greedy", "greedy_v2", "greedy_v3"}
    assert PROMPTS["baseline"] is AUDIT_SYSTEM_PROMPT
    assert PROMPTS["greedy"] is GREEDY_AUDIT_SYSTEM_PROMPT
    assert PROMPTS["greedy_v2"] is GREEDY_V2_AUDIT_SYSTEM_PROMPT
    assert PROMPTS["greedy_v3"] is GREEDY_V3_AUDIT_SYSTEM_PROMPT


def test_v2_still_has_verdict_identical_to_v1():
    # v2 changes ONLY the removed_real_evidence clause; the shared
    # still_has_verdict definition (everything before it) must be identical.
    v1_head = GREEDY_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence")[0]
    v2_head = GREEDY_V2_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence")[0]
    assert v1_head == v2_head
    # ...and the same still_has_verdict text the baseline uses.
    base_head = AUDIT_SYSTEM_PROMPT.split("removed_real_evidence")[0]
    assert v2_head == base_head


def test_v2_removed_real_evidence_differs_and_is_stricter():
    v1_tail = GREEDY_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence", 1)[1]
    v2_tail = GREEDY_V2_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence", 1)[1]
    assert v1_tail != v2_tail
    # v2 names the claim-resolution carve-out that flips the greedy-v1
    # survivors (Conclusion/Fact headers, refutation sentences) to accept.
    low = v2_tail.lower()
    assert "resolve" in low or "resolves" in low
    assert "conclusion" in low


def test_v3_still_has_verdict_identical_and_keeps_v1_carveouts():
    # v3 = v1 verbatim + carve-out (d): the shared still_has_verdict head
    # must match, v1's explicit carve-outs (a)/(b)/(c) must SURVIVE (their
    # loss was the v2 regression, run 466703), and (d) adds resolution.
    v1_head = GREEDY_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence")[0]
    v3_head = GREEDY_V3_AUDIT_SYSTEM_PROMPT.split("removed_real_evidence")[0]
    assert v1_head == v3_head
    for carveout in (
        "restates the claim being checked",
        "accurate but needs clarification",
        "summarises the reasoning behind the rating",
    ):
        assert carveout in GREEDY_V3_AUDIT_SYSTEM_PROMPT
        assert carveout in GREEDY_AUDIT_SYSTEM_PROMPT  # same v1 wording
    assert "(d)" in GREEDY_V3_AUDIT_SYSTEM_PROMPT
    assert "RESOLUTION" in GREEDY_V3_AUDIT_SYSTEM_PROMPT
    assert "(d)" not in GREEDY_AUDIT_SYSTEM_PROMPT


def test_gate_client_sends_v2_prompt():
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
        system_prompt=PROMPTS["greedy_v2"],
        post=fake_post,
    )
    client.check({"id": 1, "cleaned_doc": "text", "removed": ["span"]})
    content = captured["payload"]["messages"][0]["content"]
    assert content.startswith(GREEDY_V2_AUDIT_SYSTEM_PROMPT)
