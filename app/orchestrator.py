"""The agent loop.

Deliberately a plain, readable loop rather than a framework. Five properties
matter more than elegance here, because they are what an enterprise buyer's
security and CX teams actually ask about:

  1. bounded  - a hard step ceiling, so no conversation can spiral in cost
  2. gated    - side-effecting tools are checked by the policy plane, not the model
  3. auditable- every step is recorded with its inputs, outputs and decisions
  4. degrading- a tool error becomes a handoff, never a stack trace
  5. legible  - you can read the whole control flow on one screen
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm.base import LLMResponse
from .llm.providers import get_provider
from .memory import Session
from .policy import engine
from .prompts import system_prompt
from .tools.definitions import build_registry

# A permission. Legal in a reply only when this turn's actions include Cancel.
_CAN_BE_CANCELLED = re.compile(r"\bcan be cancelled\b", re.I)

MAX_STEPS = 6  # tool calls per user turn


@dataclass
class Turn:
    reply: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    escalated: bool = False
    # Server-computed from policy. The front end renders these and sends
    # `message` back through chat. It never decides that a click is allowed.
    actions: list[dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    def __init__(self, provider_name: str | None = None) -> None:
        self.llm = get_provider(provider_name)
        self.registry = build_registry()

    @property
    def provider_name(self) -> str:
        return self.llm.name

    def handle(self, session: Session, user_message: str) -> Turn:
        session.add("user", user_message)
        session.begin_turn()
        # A thumbs click is the customer's message, but the model does not get
        # to see it. Recording and the acknowledgement are fixed in code, so a
        # model cannot suppress the rating, invent one, or turn it into a refund.
        if engine.feedback_target(session, user_message) is not None:
            return self._record_feedback(session)
        trace: list[dict[str, Any]] = []

        for step in range(MAX_STEPS):
            response: LLMResponse = self.llm.complete(
                system=system_prompt(session),
                messages=session.working,
                tools=self.registry.schemas(),
            )

            if not response.wants_tools:
                reply = response.text or (
                    "Sorry — I lost my thread there. Could you say that again?"
                )
                return self._finish(session, reply, trace)

            # Record the model's intent, then execute each call through the
            # registry, which validates slots and injects trusted identity.
            session.working.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
            })

            for call in response.tool_calls:
                result = self.registry.execute(call.name, call.arguments, session)
                trace.append({
                    "step": step + 1,
                    "tool": call.name,
                    "args": call.arguments,
                    "status": result.get("status"),
                    # Lifted to the top of the trace entry: the outbound calls are
                    # the evidence, not a detail buried in the payload.
                    "calls": result.get("calls", []),
                    "result": _redact(result),
                })
                session.working.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    # The wire log is for the trace panel and the audit record,
                    # not for the model. Sending it would spend tokens on our
                    # own instrumentation and invite the model to reason about
                    # HTTP status codes instead of the customer.
                    "content": {k: v for k, v in result.items() if k != "calls"},
                })

        # Step ceiling hit: the model is looping. Fail safe, not silent.
        fallback = self.registry.execute(
            "escalate_to_human",
            {"reason": "repeated_failure",
             "summary": "Agent exceeded the tool-step ceiling without resolving."},
            session,
        )
        reply = (
            "I'm going round in circles here and I'd rather not waste your time — "
            f"I've passed this to a colleague (ref {fallback.get('ticket_id','pending')}) "
            "with everything we've discussed."
        )
        return self._finish(session, reply, trace)

    def _record_feedback(self, session: Session) -> Turn:
        result = self.registry.execute("record_reply_feedback", {}, session)
        trace = [{
            "step": 1,
            "tool": "record_reply_feedback",
            "args": {},
            "status": result.get("status"),
            "calls": result.get("calls", []),
            "result": _redact(result),
        }]
        if result.get("status") == "ok" and result.get("acknowledgement"):
            reply = result["acknowledgement"]
        else:
            reply = "I couldn't record that rating. Please use the thumbs under the reply."
        return self._finish(session, reply, trace)

    def _finish(self, session: Session, reply: str, trace: list[dict[str, Any]]) -> Turn:
        actions = _actions_from_trace(trace)
        reply = _bind_cancel_claim(reply, actions)
        reply_id = session.remember_reply(reply)
        # Cancel stays first, when policy allowed it. Thumbs are on every reply.
        actions.extend(engine.feedback_actions(reply_id))
        session.add("assistant", reply)
        return Turn(
            reply=reply,
            trace=trace,
            citations=list(dict.fromkeys(session.turn_citations)),
            escalated=session.escalated,
            actions=actions,
        )


def _actions_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cancel is offered only from an allowed policy decision in this turn."""
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in trace:
        for decision in _decisions(step.get("result")):
            action = engine.action_for_cancel(decision)
            if action is None or action["order_id"] in seen:
                continue
            seen.add(action["order_id"])
            actions.append(action)
    return actions


def _decisions(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if {"allowed", "code", "reason"} <= payload.keys():
            found.append(payload)
        for value in payload.values():
            found.extend(_decisions(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_decisions(item))
    return found


def _bind_cancel_claim(reply: str, actions: list[dict[str, Any]]) -> str:
    """Drop the permission if this response is not carrying the button."""
    if any(action.get("id") == "cancel" for action in actions):
        return reply
    return _CAN_BE_CANCELLED.sub("cannot be cancelled", reply)


def _redact(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the trace readable in the UI without dumping whole payloads."""
    trimmed = json.loads(json.dumps(result, default=str))
    trimmed.pop("prompt", None)
    trimmed.pop("calls", None)   # rendered separately, above the payload
    return trimmed
