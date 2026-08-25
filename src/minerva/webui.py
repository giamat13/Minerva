"""A local web chat UI for talking to a Minerva model.

Stdlib only (:mod:`http.server`), matching the project's one-dependency
ethos - a browser tab is a friendlier way to hold a conversation than a
terminal, but it does not earn Minerva a web framework. One :class:`Session`
per browser tab (keyed by a client-generated id kept in ``localStorage``)
gives real multi-turn memory; a global lock serialises generation, since one
CPU-bound model has nothing to gain from concurrent requests and every
Session's history was going to depend on request order anyway.

The effort-level selector and the "reveal reasoning" toggle are real,
generic controls on the engine-agnostic thinking scale (`thinking.py`) - not
faked for Swift. Swift's spec declares ``supports_thinking=False``, so
:meth:`~minerva.models.base.ModelSpec.resolve_thinking` always clamps the
request to ``DO`` regardless of what the dropdown asks for. Every response
reports the *resolved* level next to the requested one, so the UI stays
honest about what the loaded model can actually do instead of pretending a
reasoning trace exists when the weights never produced one.
"""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import MinervaConfig, get_config
from .errors import MinervaError
from .models.registry import load_model
from .runtime.session import Session
from .thinking import ThinkingLevel

__all__ = ["serve"]

#: The page lives in its own file, not a Python string, so it can be edited
#: (and diffed, linted, opened in a browser) like the HTML/CSS/JS it is.
#: Read fresh on every request rather than cached at import time - a small
#: file, and it means editing the page takes effect without restarting the
#: server.
_PAGE_PATH = Path(__file__).with_name("webui_chat.html")


def _level_info(level: ThinkingLevel) -> dict[str, Any]:
    return {
        "name": level.latin_name,
        "hebrew": level.hebrew_name,
        "index": int(level),
        "description": level.description,
    }


#: Where `minerva evaluate` / `minerva evaluate-instruct` write their reports.
#: The DEVDEBUG panel (see webui_chat.html) reads these fresh on every
#: request - real measured numbers, not something the UI invents. See
#: CLAUDE.md's rule on keeping them current: they are stale the moment a
#: checkpoint they describe is retrained, and a stale capability report is
#: exactly the kind of confident-but-wrong claim this whole tool exists to
#: avoid making.
_BASE_EVAL_PATH = Path("data/eval_report.json")
_INSTRUCT_EVAL_PATH = Path("data/instruct_eval_report.json")


def _stats_payload() -> dict[str, Any]:
    base = None
    if _BASE_EVAL_PATH.exists():
        d = json.loads(_BASE_EVAL_PATH.read_text(encoding="utf-8"))
        base = {
            "checkpoint": d.get("checkpoint"),
            "parameters": d.get("parameters"),
            "steps_trained": d.get("steps_trained"),
            "tokens_seen": d.get("tokens_seen"),
            "val_loss": d.get("val_loss"),
            "val_perplexity": d.get("val_perplexity"),
            "bits_per_byte": d.get("bits_per_byte"),
        }

    instruct = None
    if _INSTRUCT_EVAL_PATH.exists():
        d = json.loads(_INSTRUCT_EVAL_PATH.read_text(encoding="utf-8"))
        instruct = d.get("summary")

    return {"base": base, "instruct": instruct}


def _build_handler(model: Any, config: MinervaConfig) -> type[BaseHTTPRequestHandler]:
    sessions: dict[str, Session] = {}
    lock = threading.Lock()
    # Minted fresh per process start. The client compares it to the value it
    # last saw and clears its rendered transcript on a mismatch - otherwise a
    # server restart leaves the browser showing history the model can no
    # longer see, since `sessions` above is in-memory and restarts empty.
    instance_id = uuid.uuid4().hex

    def get_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            session = Session(model, max_history=40)
            sessions[session_id] = session
        return session

    class Handler(BaseHTTPRequestHandler):
        server_version = "MinervaChat/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
            pass

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                body = _PAGE_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/info":
                spec = model.spec
                self._send_json(
                    {
                        "model": spec.name,
                        "display_name": spec.display_name,
                        "parameter_count": spec.parameter_count,
                        "supports_thinking": spec.supports_thinking,
                        "supports_tools": spec.supports_tools,
                        "default_thinking": spec.default_thinking.latin_name,
                        "levels": [_level_info(level) for level in ThinkingLevel],
                        "instance_id": instance_id,
                    }
                )
                return
            if self.path == "/api/stats":
                self._send_json(_stats_payload())
                return
            self._send_json({"error": f"no such route: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path == "/api/chat":
                self._handle_chat()
                return
            if self.path == "/api/reset":
                data = self._read_json()
                with lock:
                    sessions.pop(data.get("session_id", ""), None)
                self._send_json({"ok": True})
                return
            self._send_json({"error": f"no such route: {self.path}"}, status=404)

        def _handle_chat(self) -> None:
            data = self._read_json()
            session_id = str(data.get("session_id") or "")
            message = str(data.get("message") or "").strip()
            requested = data.get("thinking") or None
            if not session_id or not message:
                self._send_json({"error": "session_id and message are required"}, status=400)
                return

            try:
                with lock:
                    session = get_session(session_id)
                    run = session.send_run(message, thinking=requested)
            except MinervaError as exc:
                self._send_json({"error": str(exc)}, status=502)
                return
            except Exception as exc:  # a real, unexpected failure - say so plainly
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return

            tool_results = [tr for step in run.steps for tr in step.tool_results]
            tool_calls = [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": result.content,
                    "is_error": result.is_error,
                }
                for call, result in zip(run.tool_calls, tool_results, strict=True)
            ]

            self._send_json(
                {
                    "answer": run.answer,
                    "thinking": run.thinking,
                    "requested_thinking": requested,
                    "effective_thinking": _level_info(run.thinking_level),
                    "tool_calls": tool_calls,
                    "duration_seconds": round(run.duration_seconds, 2),
                }
            )

    return Handler


def serve(
    *,
    model_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8420,
    config: MinervaConfig | None = None,
) -> None:
    """Load a model and run the chat UI until interrupted."""
    config = config or get_config()
    model = load_model(model_name, config=config)

    handler = _build_handler(model, config)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Minerva {model.spec.display_name} - chat UI at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

