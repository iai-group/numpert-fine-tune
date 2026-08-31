"""LLM client for NLI scoring against a vLLM OpenAI-compatible endpoint.

The vLLM server must be running before any scoring is attempted (see
``slurm/run_vllm_gemma_nli.sh``). Structured output is requested via the
``response_format`` JSON-schema field (guided decoding). gemma's chat template
rejects a standalone system role, so ``system_in_user=True`` (the default)
prepends the system text to the user message.
"""

import json
from typing import Any, Callable, Dict, List, Optional

import requests

from src.number_annotation.core.schema import NumberSpan
from src.annotation_prompt.prompt import (
    NLI_SYSTEM_PROMPT,
    build_nli_user_prompt,
)
from src.number_annotation.nli.schema import (
    NLI_OUTPUT_SCHEMA,
    NliItem,
    parse_nli_response,
)

DEFAULT_NLI_MODEL = "google/gemma-4-12B-it"
DEFAULT_NLI_BASE_URL = "http://localhost:8000/v1"
# One call scores every number in a sentence window, so a dense sentence can
# emit many result objects (each with a free-text rationale). 512 truncated
# real inputs (finish_reason=length); 2048 leaves ample headroom under the
# 4096 max-model-len while inputs stay short.
DEFAULT_NLI_MAX_TOKENS = 2048


class NliClient:
    """Score one sentence window's numbers via a single vLLM chat call."""

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        base_url: str = DEFAULT_NLI_BASE_URL,
        max_tokens: int = DEFAULT_NLI_MAX_TOKENS,
        temperature: float = 0.0,
        timeout: int = 300,
        system_in_user: bool = True,
        post: Callable[..., Any] = requests.post,
    ):
        """Store inference parameters and the injectable HTTP poster."""
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._system_in_user = system_in_user
        self._post = post

    def _messages(self, user_content: str) -> List[Dict[str, str]]:
        """Build the chat messages, folding system into user for gemma."""
        if self._system_in_user:
            return [
                {
                    "role": "user",
                    "content": f"{NLI_SYSTEM_PROMPT}\n\n{user_content}",
                }
            ]
        return [
            {"role": "system", "content": NLI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def score(
        self,
        claim: str,
        claim_numbers: List[NumberSpan],
        marked_sentence: str,
    ) -> List[NliItem]:
        """Score every marked number in the sentence against the claim."""
        user_content = build_nli_user_prompt(
            claim, claim_numbers, marked_sentence
        )
        payload = {
            "model": self._model,
            "messages": self._messages(user_content),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nli",
                    "strict": True,
                    "schema": NLI_OUTPUT_SCHEMA,
                },
            },
        }
        resp = self._post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "vLLM NLI response truncated at max_tokens="
                f"{self._max_tokens} (finish_reason=length). Increase "
                "max_tokens."
            )
        content = choice["message"]["content"]
        return parse_nli_response(json.loads(content))


class FakeNliClient:
    """Deterministic test double: canned items per sentence, else default."""

    def __init__(
        self,
        mapping: Optional[Dict[str, List[NliItem]]] = None,
        default: Optional[List[NliItem]] = None,
    ):
        """Store an optional per-sentence mapping and a fallback list."""
        self._mapping = mapping or {}
        self._default = default or []

    def score(
        self,
        claim: str,
        claim_numbers: List[NumberSpan],
        marked_sentence: str,
    ) -> List[NliItem]:
        """Return canned items for the sentence (or the default)."""
        return list(self._mapping.get(marked_sentence, self._default))
