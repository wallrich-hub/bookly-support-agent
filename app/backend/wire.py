"""Egress log: every call the agent makes to a system, and what came back.

Two audiences, one mechanism.

Operationally this is the record of what the agent actually did to your systems -
the thing you want when a customer disputes a refund, and the thing a security
review asks for first.

In the demo it is the proof. "We call real APIs" is a claim; a POST to
api.stripe.com with an idempotency key and a 200 carrying `re_...` is evidence.
The in-memory backends record here too, deliberately, so the trace looks the
same shape in both modes and the difference between them is visible rather than
described.

Never records: Authorization headers, API keys, card data. Bodies are truncated.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar("wire_calls", default=None)

MAX_PREVIEW = 400
REDACTED_KEYS = {"authorization", "api_key", "secret", "password", "card", "cvc"}


def begin() -> None:
    """Start capturing. Called once per tool execution."""
    _CALLS.set([])


def drain() -> list[dict[str, Any]]:
    """Take everything captured since begin() and reset."""
    calls = _CALLS.get() or []
    _CALLS.set([])
    return calls


def _record(entry: dict[str, Any]) -> None:
    calls = _CALLS.get()
    if calls is not None:
        calls.append(entry)


def _safe(params: Any) -> Any:
    """Drop anything that must never appear in a log or on a screen."""
    if not isinstance(params, dict):
        return params
    return {
        k: ("«redacted»" if any(r in k.lower() for r in REDACTED_KEYS) else v)
        for k, v in params.items()
    }


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text[:MAX_PREVIEW] + ("…" if len(text) > MAX_PREVIEW else "")


class span:
    """Context manager timing one outbound call.

        with wire.span("stripe", "POST", "/refunds", params=form) as call:
            ...
            call.done(status=200, summary="re_123 succeeded")
    """

    def __init__(self, system: str, method: str, path: str,
                 params: Any = None, transport: str = "http") -> None:
        self.entry: dict[str, Any] = {
            "system": system,
            "transport": transport,       # http | local
            "method": method,
            "path": path,
            "params": _safe(params) if params else None,
            "status": None,
            "outcome": "pending",
            "summary": "",
            "ms": 0.0,
        }
        self._t0 = 0.0

    def __enter__(self) -> "span":
        self._t0 = time.perf_counter()
        return self

    def done(self, status: int | None = None, summary: str = "") -> None:
        self.entry["status"] = status
        self.entry["summary"] = _preview(summary)

    def note(self, **fields: Any) -> None:
        self.entry.update(fields)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.entry["ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        if exc is not None:
            self.entry["outcome"] = "error"
            self.entry["summary"] = _preview(f"{exc_type.__name__}: {exc}")
        elif self.entry["outcome"] == "pending":
            code = self.entry["status"]
            self.entry["outcome"] = (
                "ok" if code is None or code < 400 else "error"
            )
        _record(self.entry)
        return False  # never swallow


def local(system: str, operation: str, summary: str = "") -> span:
    """A call served from in-process data rather than the network."""
    return span(system, "CALL", operation, transport="local")
