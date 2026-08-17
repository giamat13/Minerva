"""A real Ollama engine.

This talks to a live Ollama daemon over its HTTP API - no simulation, no
canned responses. If no daemon is running, calls raise
:class:`~minerva.errors.EngineUnavailableError` rather than pretending.

Endpoints used
--------------
``GET  /api/version``  - health probe
``GET  /api/tags``     - installed models
``POST /api/show``     - model metadata (capabilities, parameter count)
``POST /api/chat``     - generation, with tools and thinking
``POST /api/pull``     - install a model (streaming progress)

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import httpx

from ..errors import (
    EngineRequestError,
    EngineResponseError,
    EngineUnavailableError,
)
from ..messages import Message, Role, ToolCall
from ..thinking import ThinkingProfile
from .base import (
    Engine,
    EngineCapabilities,
    EngineHealth,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
    Usage,
)

__all__ = ["DEFAULT_OLLAMA_HOST", "OllamaEngine"]

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Ollama exposes decoding parameters under "options" with its own names.
# Left-hand side is Minerva's neutral name, right-hand side is Ollama's.
_OPTION_NAMES: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "max_tokens": "num_predict",
    "context_length": "num_ctx",
    "stop": "stop",
    "seed": "seed",
}


class OllamaEngine(Engine):
    """Inference through a locally running Ollama daemon."""

    name = "ollama"
    capabilities = EngineCapabilities(
        streaming=True,
        tools=True,
        thinking=True,
        thinking_trace=True,
        # Ollama takes `think: true|false|"low"|"medium"|"high"`, never a
        # numeric budget, so budget_tokens from a ThinkingProfile is ignored.
        thinking_budget=False,
        vision=True,
    )

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout: float = 300.0,
        connect_timeout: float = 5.0,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        :param host: Base URL of the daemon, e.g. ``http://127.0.0.1:11434``.
        :param timeout: Read timeout in seconds. Generous by default because a
            cold model load on CPU genuinely can take minutes.
        :param connect_timeout: Connect timeout. Short, so "nothing is
            listening" is reported immediately instead of hanging.
        :param client: Inject your own ``httpx.Client`` (proxies, TLS, auth).
        :param headers: Extra headers, e.g. an ``Authorization`` header for a
            reverse-proxied remote daemon.
        """
        self.host = host.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.host,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers={"content-type": "application/json", **(headers or {})},
        )

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()

    # -- health & inventory ----------------------------------------------

    def health(self) -> EngineHealth:
        """Probe the daemon. Reports failure instead of raising, by contract."""
        try:
            response = self._client.get("/api/version", timeout=httpx.Timeout(5.0, connect=2.0))
            response.raise_for_status()
            version = response.json().get("version")
        except httpx.HTTPError as exc:
            return EngineHealth(
                available=False,
                endpoint=self.host,
                detail=f"{type(exc).__name__}: {exc}",
            )
        except (ValueError, KeyError) as exc:
            return EngineHealth(
                available=False,
                endpoint=self.host,
                detail=f"unexpected /api/version payload: {exc}",
            )

        try:
            models = tuple(self.list_available_models())
        except Exception:
            models = ()

        return EngineHealth(available=True, endpoint=self.host, version=version, models=models)

    def list_available_models(self) -> list[str]:
        """Every model tag installed on the daemon (e.g. ``["qwen3:1.7b"]``)."""
        payload = self._get_json("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise EngineResponseError("/api/tags did not return a 'models' list")
        return [
            str(entry["model"])
            for entry in models
            if isinstance(entry, dict) and "model" in entry
        ]

    def show_model(self, model: str) -> dict[str, Any]:
        """Raw metadata for an installed model (family, sizes, capabilities)."""
        return self._post_json("/api/show", {"model": model})

    def model_capabilities(self, model: str) -> list[str]:
        """The capability tags Ollama reports, e.g. ``["completion", "tools", "thinking"]``.

        Useful for checking whether a specific weight file supports tool
        calling before you send it tools.
        """
        payload = self.show_model(model)
        caps = payload.get("capabilities", [])
        return [str(cap) for cap in caps] if isinstance(caps, list) else []

    def pull(self, model: str) -> Iterator[dict[str, Any]]:
        """Download a model, yielding Ollama's real progress events.

        This is a generator: nothing is downloaded until you iterate it.
        """
        try:
            with self._client.stream(
                "POST",
                "/api/pull",
                json={"model": model, "stream": True},
                timeout=httpx.Timeout(None, connect=5.0),
            ) as response:
                self._raise_for_status(response, read_body=True)
                for line in response.iter_lines():
                    if line.strip():
                        yield json.loads(line)
        except httpx.ConnectError as exc:
            raise EngineUnavailableError(self._unavailable_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineRequestError(f"pulling {model!r} failed: {exc}") from exc

    # -- generation ------------------------------------------------------

    def chat(self, request: GenerationRequest) -> GenerationResult:
        """One non-streaming generation."""
        payload = self._build_payload(request, stream=False)
        try:
            data = self._post_json("/api/chat", payload)
        except EngineRequestError as exc:
            if not _is_think_value_rejection(exc, payload):
                raise
            data = self._post_json("/api/chat", _downgrade_think(payload))
        return self._decode_result(data, fallback_model=request.model)

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """One streaming generation.

        Ollama streams newline-delimited JSON. Content and reasoning arrive as
        separate fields on each frame, so we can surface them on separate
        channels; tool calls only ever arrive complete, never token by token.
        """
        payload = self._build_payload(request, stream=True)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final_frame: dict[str, Any] = {}

        try:
            with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(None, connect=5.0),
            ) as response:
                try:
                    self._raise_for_status(response, read_body=True)
                except EngineRequestError as exc:
                    if not _is_think_value_rejection(exc, payload):
                        raise
                    # Nothing has been yielded yet, so retrying here is safe.
                    yield from self.stream(
                        replace(request, extra={**request.extra, "think": True})
                    )
                    return
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise EngineResponseError(
                            f"malformed JSON frame from /api/chat: {line[:200]!r}"
                        ) from exc

                    if error := frame.get("error"):
                        raise EngineRequestError(f"ollama error: {error}")

                    message = frame.get("message") or {}

                    if thinking := message.get("thinking"):
                        thinking_parts.append(thinking)
                        yield StreamChunk(kind="thinking", text=thinking, raw=frame)

                    if content := message.get("content"):
                        content_parts.append(content)
                        yield StreamChunk(kind="content", text=content, raw=frame)

                    for call in self._decode_tool_calls(message.get("tool_calls")):
                        tool_calls.append(call)
                        yield StreamChunk(kind="tool_call", tool_call=call, raw=frame)

                    if frame.get("done"):
                        final_frame = frame
        except httpx.ConnectError as exc:
            raise EngineUnavailableError(self._unavailable_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineRequestError(f"streaming request to /api/chat failed: {exc}") from exc

        assembled = Message(
            role=Role.ASSISTANT,
            content="".join(content_parts),
            thinking="".join(thinking_parts) or None,
            tool_calls=tool_calls,
        )
        result = GenerationResult(
            message=assembled,
            model=str(final_frame.get("model", request.model)),
            finish_reason=final_frame.get("done_reason"),
            usage=_decode_usage(final_frame),
            duration_seconds=_nanos_to_seconds(final_frame.get("total_duration")),
            raw=final_frame,
        )
        yield StreamChunk(kind="done", result=result, raw=final_frame)

    # -- payload construction --------------------------------------------

    def _build_payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._encode_message(m) for m in request.messages],
            "stream": stream,
        }

        if request.tools:
            payload["tools"] = list(request.tools)

        think = self._encode_thinking(request.thinking)
        if think is not None:
            payload["think"] = think

        options = self._encode_options(request.sampling)
        if options:
            payload["options"] = options

        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive

        # Engine-specific escape hatch, applied last so it can override.
        payload.update(request.extra)
        return payload

    @staticmethod
    def _encode_thinking(profile: ThinkingProfile | None) -> bool | str | None:
        """Translate a solfege thinking profile into Ollama's ``think`` field.

        Ollama accepts a boolean for most reasoning models and a coarse
        ``"low"``/``"medium"``/``"high"`` string for models that expose an
        effort knob (gpt-oss). We send the effort string when we have one: on
        models that only understand the boolean, Ollama treats any non-empty
        effort as "thinking on", so the meaning is preserved either way.
        """
        if profile is None:
            return None
        if not profile.enabled:
            return False
        return profile.effort or True

    @staticmethod
    def _encode_options(sampling: SamplingParams) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for neutral, ollama_name in _OPTION_NAMES.items():
            value = getattr(sampling, neutral, None)
            if value is None or value == ():
                continue
            options[ollama_name] = list(value) if isinstance(value, tuple) else value
        return options

    @staticmethod
    def _encode_message(message: Message) -> dict[str, Any]:
        """Translate a Minerva message into an Ollama chat message."""
        data: dict[str, Any] = {"role": message.role.value, "content": message.content}

        # Extended Thinking: sending the reasoning trace back lets the model
        # continue from where it left off instead of re-deriving it.
        if message.role is Role.ASSISTANT and message.thinking:
            data["thinking"] = message.thinking

        if message.tool_calls:
            data["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]

        if message.role is Role.TOOL:
            # Ollama matches tool results to calls by name.
            if message.name:
                data["tool_name"] = message.name
            if message.tool_call_id:
                data["tool_call_id"] = message.tool_call_id

        return data

    # -- response decoding -----------------------------------------------

    def _decode_result(self, data: dict[str, Any], *, fallback_model: str) -> GenerationResult:
        raw_message = data.get("message")
        if not isinstance(raw_message, dict):
            raise EngineResponseError(
                f"/api/chat response has no 'message' object; got keys: {sorted(data)}"
            )

        message = Message(
            role=Role.ASSISTANT,
            content=str(raw_message.get("content") or ""),
            thinking=raw_message.get("thinking") or None,
            tool_calls=self._decode_tool_calls(raw_message.get("tool_calls")),
        )
        return GenerationResult(
            message=message,
            model=str(data.get("model", fallback_model)),
            finish_reason=data.get("done_reason"),
            usage=_decode_usage(data),
            duration_seconds=_nanos_to_seconds(data.get("total_duration")),
            raw=data,
        )

    @staticmethod
    def _decode_tool_calls(raw: Any) -> list[ToolCall]:
        """Turn Ollama's ``tool_calls`` into :class:`ToolCall` objects.

        Ollama normally returns ``arguments`` already decoded as an object, but
        some model templates emit it as a JSON string. We accept both so a
        template quirk never breaks a tool call.
        """
        if not raw:
            return []
        if not isinstance(raw, list):
            raise EngineResponseError(f"'tool_calls' should be a list, got {type(raw).__name__}")

        calls: list[ToolCall] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            function = entry.get("function") or {}
            name = function.get("name")
            if not name:
                continue

            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    raise EngineResponseError(
                        f"tool call {name!r} has arguments that are neither an object "
                        f"nor valid JSON: {arguments[:200]!r}"
                    ) from exc
            if not isinstance(arguments, dict):
                raise EngineResponseError(
                    f"tool call {name!r} has non-object arguments: {type(arguments).__name__}"
                )

            call_id = entry.get("id")
            extra = {"id": str(call_id)} if call_id else {}
            calls.append(ToolCall(name=str(name), arguments=arguments, **extra))
        return calls

    # -- HTTP plumbing ---------------------------------------------------

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = self._client.get(path)
            self._raise_for_status(response)
            return self._json(response)
        except httpx.ConnectError as exc:
            raise EngineUnavailableError(self._unavailable_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineRequestError(f"GET {path} failed: {exc}") from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
            self._raise_for_status(response)
            return self._json(response)
        except httpx.ConnectError as exc:
            raise EngineUnavailableError(self._unavailable_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EngineRequestError(f"POST {path} failed: {exc}") from exc

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise EngineResponseError(
                f"{response.request.method} {response.request.url.path} returned "
                f"non-JSON body: {response.text[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise EngineResponseError(
                f"expected a JSON object from {response.request.url.path}, "
                f"got {type(data).__name__}"
            )
        return data

    @staticmethod
    def _raise_for_status(response: httpx.Response, *, read_body: bool = False) -> None:
        if response.status_code < 400:
            return
        if read_body:
            response.read()
        detail = response.text.strip()
        try:
            parsed = json.loads(detail)
            detail = parsed.get("error", detail) if isinstance(parsed, dict) else detail
        except ValueError:
            pass
        raise EngineRequestError(
            f"ollama returned HTTP {response.status_code}: {detail[:400] or '(empty body)'}",
            status_code=response.status_code,
        )

    def _unavailable_message(self, exc: Exception) -> str:
        return (
            f"cannot reach Ollama at {self.host} ({exc}). "
            f"Start it with `ollama serve`, or point Minerva elsewhere with "
            f"MINERVA_OLLAMA_HOST."
        )


# ---------------------------------------------------------------------------
# Thinking-value compatibility
#
# Ollama's `think` field accepts a boolean on every reasoning-capable model,
# but the coarse "low"/"medium"/"high" effort string only on models that expose
# an effort knob (gpt-oss and friends). Which of those a given daemon build
# accepts for a given model is not discoverable up front, so rather than
# guessing conservatively and throwing away the finer signal, we send the
# effort string and fall back to `think: true` if - and only if - the daemon
# rejects that specific field. The user's intent ("think harder") survives
# either way.
# ---------------------------------------------------------------------------


def _is_think_value_rejection(exc: EngineRequestError, payload: dict[str, Any]) -> bool:
    """True when this failure is the daemon refusing our effort string.

    Deliberately narrow: only a 4xx, only when we actually sent an effort
    string, and only when the error text points at the thinking field. Any
    other failure must propagate untouched - a blanket retry would paper over
    real errors.
    """
    if exc.status_code is None or not (400 <= exc.status_code < 500):
        return False
    if not isinstance(payload.get("think"), str):
        return False
    message = str(exc).lower()
    return "think" in message or "reasoning" in message


def _downgrade_think(payload: dict[str, Any]) -> dict[str, Any]:
    """The same payload with the effort string replaced by a plain ``true``."""
    return {**payload, "think": True}


def _decode_usage(data: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=_as_int(data.get("prompt_eval_count")),
        completion_tokens=_as_int(data.get("eval_count")),
    )


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _nanos_to_seconds(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    return value / 1e9 if isinstance(value, (int, float)) else None


def format_pull_progress(events: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Turn ``/api/pull`` events into human-readable progress lines."""
    for event in events:
        status = event.get("status", "")
        total, completed = event.get("total"), event.get("completed")
        if isinstance(total, (int, float)) and isinstance(completed, (int, float)) and total:
            yield (
                f"{status}: {completed / total * 100:5.1f}% "
                f"({completed / 1e6:.0f}/{total / 1e6:.0f} MB)"
            )
        elif status:
            yield status


def encode_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Public helper: encode messages exactly as :class:`OllamaEngine` would.

    Exposed so tests and debugging tools can inspect the real wire payload
    without reaching into private methods.
    """
    return [OllamaEngine._encode_message(message) for message in messages]
