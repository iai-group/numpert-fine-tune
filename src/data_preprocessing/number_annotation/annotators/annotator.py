"""Annotator protocol and implementations.

`parse_response` and `FakeAnnotator` are deterministic and unit-tested.
`ClaudeCodeAnnotator` drives Claude Code headlessly (`claude -p ...
--json-schema ...`) using the user's Claude Code subscription auth (OAuth
/ keychain) — no API key. We deliberately do NOT pass `--bare`: bare mode
skips OAuth/keychain and requires an API key, so it breaks subscription
(incl. team/non-admin) auth. To keep per-call overhead low without
`--bare`, we block MCP autodiscovery (`--strict-mcp-config --mcp-config
{}`) and run the CLI from a neutral working directory so no project
context (CLAUDE.md / .mcp.json / .claude/) is auto-loaded. The model alias
is configurable; the default frontier alias is `opus`. (Open / non-Claude
models go through an API in Phase 2, not here.)

`VertexAnnotator` calls the Anthropic API directly via Google Vertex AI,
authenticating with gcloud application-default credentials
(`gcloud auth application-default login`). Requires `pip install
'anthropic[vertex]'`. Models use Vertex IDs such as
`claude-opus-4-5@20251101`.
"""

import json
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Protocol

import requests

from src.number_annotation.core.prompts import (
    ANNOTATION_OUTPUT_SCHEMA,
    ANNOTATION_SYSTEM_PROMPT,
    build_annotation_user_prompt,
)
from src.number_annotation.core.schema import RawNumber, raw_from_dict

DEFAULT_ANNOTATOR_MODEL = "opus"
DEFAULT_VERTEX_MODEL = "gemini-2.5-flash"
DEFAULT_VERTEX_REGION = "europe-west1"
# Long source documents can contain dozens of numeric mentions; the resulting
# annotation JSON can exceed 4096 tokens and get truncated (finish_reason=
# "length"), producing unparseable output. 12288 leaves room for these, paired
# with a --max-model-len of >= ~16k on the vLLM server.
DEFAULT_MAX_TOKENS = 12288
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_REGION = "europe-west1"
DEFAULT_GEMINI_MAX_TOKENS = 8192


class Annotator(Protocol):
    """Anything that can turn text into a list of RawNumber."""

    def annotate(self, text: str) -> List[RawNumber]:
        """Return the numeric mentions found in `text`."""
        ...


def parse_response(data: Dict[str, Any]) -> List[RawNumber]:
    """Parse a structured-output payload into RawNumber objects."""
    return [raw_from_dict(item) for item in data.get("numbers", [])]


class FakeAnnotator:
    """Deterministic test double mapping exact text to raw numbers."""

    def __init__(self, mapping: Dict[str, List[RawNumber]]):
        """Store the text -> raw-numbers mapping."""
        self._mapping = mapping

    def annotate(self, text: str) -> List[RawNumber]:
        """Return the mapped raw numbers, or an empty list."""
        return list(self._mapping.get(text, []))


class ClaudeCodeAnnotator:
    """Annotator that drives Claude Code headlessly via `claude -p`."""

    def __init__(
        self,
        model: str = DEFAULT_ANNOTATOR_MODEL,
        binary: str = "claude",
        runner: Callable[..., Any] = subprocess.run,
    ):
        """Store the model alias, CLI binary, and subprocess runner.

        `runner` is injectable so tests never invoke the real CLI.
        """
        self._model = model
        self._binary = binary
        self._runner = runner

    def build_command(self) -> List[str]:
        """Build the `claude -p` command with structured-output flags.

        No `--bare` (it would break subscription/OAuth auth). MCP
        autodiscovery is blocked to keep calls lean.
        """
        return [
            self._binary,
            "-p",
            "--model",
            self._model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ANNOTATION_OUTPUT_SCHEMA),
            "--append-system-prompt",
            ANNOTATION_SYSTEM_PROMPT,
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
        ]

    def annotate(self, text: str) -> List[RawNumber]:
        """Annotate `text` by invoking Claude Code and parsing JSON.

        The full user prompt (instructions + text) is passed on stdin;
        Claude Code returns a JSON envelope whose `structured_output`
        field is the schema-validated annotation payload. The CLI runs
        from a neutral temp dir so no project context is auto-loaded.
        """
        result = self._runner(
            self.build_command(),
            input=build_annotation_user_prompt(text),
            capture_output=True,
            text=True,
            check=True,
            cwd=tempfile.gettempdir(),
        )
        payload = json.loads(result.stdout)
        return parse_response(payload["structured_output"])


class VertexAnnotator:
    """Annotator that calls Claude on Vertex AI via gcloud default auth.

    Requires ``pip install 'anthropic[vertex]'`` and valid application-
    default credentials (``gcloud auth application-default login``).

    Temperature is silently ignored for models that don't accept it
    (Opus 4.7+); pass ``None`` (default) to omit it entirely.
    """

    def __init__(
        self,
        model: str = DEFAULT_VERTEX_MODEL,
        project_id: Optional[str] = None,
        region: str = DEFAULT_VERTEX_REGION,
        temperature: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialise client and store inference parameters."""
        try:
            from anthropic import AnthropicVertex  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "anthropic[vertex] is required: pip install 'anthropic[vertex]'"
            ) from exc

        self._client = AnthropicVertex(
            region=region,
            project_id=project_id,  # None → inferred from gcloud config
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def annotate(self, text: str) -> List[RawNumber]:
        """Annotate ``text`` via the Anthropic Messages API on Vertex AI."""
        kwargs: Dict[str, Any] = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=ANNOTATION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_annotation_user_prompt(text),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": ANNOTATION_OUTPUT_SCHEMA,
                }
            },
            **kwargs,
        )
        text_block = next(b for b in response.content if b.type == "text")
        return parse_response(json.loads(text_block.text))


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON Schema to Gemini-compatible schema.

    Gemini does not support ``anyOf`` for nullable fields; converts
    ``anyOf: [{type: X}, {type: "null"}]`` to ``{type: X, nullable: true}``.
    """
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            result = dict(non_null[0])
            result["nullable"] = True
            return result
        return dict(schema)

    result = {k: v for k, v in schema.items()}
    if "properties" in schema:
        result["properties"] = {
            k: _to_gemini_schema(v) for k, v in schema["properties"].items()
        }
    if "items" in schema:
        result["items"] = _to_gemini_schema(schema["items"])
    return result


class GeminiAnnotator:
    """Annotator that calls Gemini on Vertex AI via gcloud default auth.

    Requires ``pip install google-genai`` and valid application-default
    credentials (``gcloud auth application-default login``).
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        project_id: Optional[str] = None,
        region: str = DEFAULT_GEMINI_REGION,
        max_tokens: int = DEFAULT_GEMINI_MAX_TOKENS,
    ):
        """Initialise Vertex AI client and store inference parameters."""
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types as gtypes  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai is required: pip install google-genai"
            ) from exc

        self._client = genai.Client(
            vertexai=True,
            project=project_id,  # None → inferred from gcloud config
            location=region,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._gtypes = gtypes

    def annotate(self, text: str) -> List[RawNumber]:
        """Annotate ``text`` via the Gemini generate-content API on Vertex AI."""
        schema = _to_gemini_schema(ANNOTATION_OUTPUT_SCHEMA)
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_annotation_user_prompt(text),
            config=self._gtypes.GenerateContentConfig(
                system_instruction=ANNOTATION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=self._max_tokens,
                # Disable thinking: annotation is deterministic extraction,
                # not reasoning.  Without this, thinking tokens consume most
                # of the output budget and truncate long responses.
                thinking_config=self._gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )
        return parse_response(json.loads(response.text))


class VllmAnnotator:
    """Annotator that calls a vLLM OpenAI-compatible HTTP server.

    The server must be running before any annotation is attempted.  Start it
    with::

        vllm serve "Qwen/Qwen3.6-35B-A3B" --port 8000

    Structured output is requested via the ``response_format`` JSON-schema
    field (vLLM guided-decoding), so the model is constrained to emit valid
    annotation JSON.
    """

    def __init__(
        self,
        model: str = DEFAULT_VLLM_MODEL,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        # A single request can generate up to ``max_tokens`` (12288) of JSON.
        # On GPUs without nvcc, vLLM runs in eager mode (no CUDA graphs) at only
        # ~9 tok/s, so a full 12288-token response can take ~22 min. 1800 s
        # (30 min) covers that worst case; well beyond the original 120 s.
        timeout: int = 1800,
    ):
        """Store inference parameters and construct the endpoint URL."""
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def annotate(self, text: str) -> List[RawNumber]:
        """Annotate ``text`` via the vLLM chat-completions endpoint."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_annotation_user_prompt(text),
                },
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "annotation",
                    "strict": True,
                    "schema": ANNOTATION_OUTPUT_SCHEMA,
                },
            },
        }
        resp = requests.post(
            self._url,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        content = choice["message"]["content"]
        # A truncated response (finish_reason == "length") yields incomplete
        # JSON that json.loads cannot parse. Surface a clear, actionable error
        # instead of an opaque JSONDecodeError so the cause (max_tokens too low
        # for a long document) is obvious from the logs.
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "vLLM response truncated at max_tokens="
                f"{self._max_tokens} (finish_reason=length); the annotation "
                "JSON is incomplete. Increase max_tokens (and --max-model-len) "
                "to fit long documents."
            )
        return parse_response(json.loads(content))
