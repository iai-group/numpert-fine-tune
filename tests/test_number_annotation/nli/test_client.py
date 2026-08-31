"""Tests for the NLI vLLM client (mocked HTTP) and the fake client."""

import json

import pytest

from src.number_annotation.nli.client import (
    DEFAULT_NLI_MAX_TOKENS,
    DEFAULT_NLI_MODEL,
    FakeNliClient,
    NliClient,
)
from src.number_annotation.nli.schema import NliItem


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _ok_payload(finish_reason="stop"):
    content = json.dumps(
        {"results": [
            {"marker": "N0", "relevance": 5, "relation": "supports",
             "claim_num": 0, "rationale": "ok"},
        ]}
    )
    return {"choices": [
        {"message": {"content": content}, "finish_reason": finish_reason}
    ]}


def test_score_parses_items_and_builds_payload():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp(_ok_payload())

    client = NliClient(post=fake_post)
    items = client.score("A claim.", [], "Sentence [N0]40%[/N0].")
    assert items == [NliItem("N0", 5, "supports", 0, "ok")]
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["model"] == DEFAULT_NLI_MODEL
    assert captured["json"]["response_format"]["type"] == "json_schema"


def test_system_in_user_folds_system_into_user_message():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _Resp(_ok_payload())

    client = NliClient(post=fake_post, system_in_user=True)
    client.score("A claim.", [], "Sentence [N0]40%[/N0].")
    roles = [m["role"] for m in captured["json"]["messages"]]
    assert roles == ["user"]  # no standalone system role for gemma


def test_standalone_system_role_when_disabled():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _Resp(_ok_payload())

    client = NliClient(post=fake_post, system_in_user=False)
    client.score("A claim.", [], "Sentence [N0]40%[/N0].")
    roles = [m["role"] for m in captured["json"]["messages"]]
    assert roles == ["system", "user"]


def test_truncated_response_raises():
    def fake_post(url, json=None, timeout=None):
        return _Resp(_ok_payload(finish_reason="length"))

    client = NliClient(post=fake_post)
    with pytest.raises(RuntimeError, match="truncated"):
        client.score("A claim.", [], "Sentence [N0]40%[/N0].")


def test_default_max_tokens_has_headroom_for_many_numbers():
    # 512 truncated real multi-number sentences; the default must be larger.
    assert DEFAULT_NLI_MAX_TOKENS >= 2048


def test_max_tokens_is_configurable_in_payload():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _Resp(_ok_payload())

    client = NliClient(post=fake_post, max_tokens=4096)
    client.score("A claim.", [], "Sentence [N0]40%[/N0].")
    assert captured["json"]["max_tokens"] == 4096


def test_fake_client_returns_canned_items():
    canned = [NliItem("N0", 3, "neutral", None, "x")]
    fake = FakeNliClient(default=canned)
    assert fake.score("any", [], "any") == canned
