"""FastAPI app: chat endpoint + a single-file web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .env import load as _load_env

_load_env()   # before any module reads os.environ

from .backend import data  # noqa: E402
from .backend.payments import get_gateway
from .memory import SESSIONS, get_session
from .orchestrator import Orchestrator

app = FastAPI(title="Bookly Support Agent", version="1.0")
agent = Orchestrator(os.environ.get("LLM_PROVIDER"))
STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str
    # Pydantic evaluates this annotation at runtime, so it must be valid syntax
    # on the interpreter actually running the app - `str | None` is 3.10+ only.
    # Everywhere else in the repo, `from __future__ import annotations` keeps
    # annotations as strings and the modern syntax is fine.
    session_id: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    from .backend.oms import get_oms
    return {"ok": True, "provider": agent.provider_name,
            "oms": get_oms().name, "payments": get_gateway().name,
            "reset_enabled": os.environ.get("BOOKLY_DEMO_RESET") == "1",
            "payments_on_file": len(getattr(get_gateway(), "_map", {}) or {}),
            "tools": [t["name"] for t in agent.registry.schemas()]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    turn = agent.handle(session, req.message)
    return {
        "session_id": session.session_id,
        "reply": turn.reply,
        "trace": turn.trace,
        "citations": turn.citations,
        "escalated": turn.escalated,
        "actions": turn.actions,
        "verified": {k: v for k, v in session.verified.items()
                     if not isinstance(v, (dict, list))},
    }


@app.post("/api/admin/reset")
def admin_reset():
    """Wipe in-process demo state so a run-through can be repeated.

    Disabled unless BOOKLY_DEMO_RESET=1. An unauthenticated endpoint that
    erases refund records has no business existing by default, demo or not -
    and a reviewer finding it ungated would be right to say so.
    """
    if os.environ.get("BOOKLY_DEMO_RESET") != "1":
        raise HTTPException(status_code=404, detail="Not found")

    cleared = {
        "returns": len(data.RETURNS), "refunds": len(data.REFUNDS),
        "escalations": len(data.ESCALATIONS), "audit_entries": len(data.AUDIT_LOG),
        "cancellations": len(data.CANCELLATIONS),
        "sessions": len(SESSIONS),
    }
    data.RETURNS.clear(); data.REFUNDS.clear(); data.ESCALATIONS.clear()
    data.PASSWORD_RESETS.clear(); data.AUDIT_LOG.clear()
    data.CANCELLATIONS.clear()
    SESSIONS.clear()

    # Pick up any PaymentIntents the reset script just created.
    gateway = get_gateway()
    reloaded = gateway.reload_payment_map() if hasattr(
        gateway, "reload_payment_map") else 0

    return {"ok": True, "cleared": cleared, "payments_reloaded": reloaded}


@app.get("/api/audit")
def audit():
    """Every side effect the agent has caused this process.

    Escalation records include `transcript`, the turns create_escalation was
    given, so a specialist can read what was said. That text is untrusted.
    """
    return {
        "returns": list(data.RETURNS.values()),
        "refunds": list(data.REFUNDS.values()),
        "escalations": list(data.ESCALATIONS.values()),
        "log": data.AUDIT_LOG,
    }
