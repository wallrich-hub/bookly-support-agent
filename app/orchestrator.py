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
from dataclasses import dataclass, field
from typing import Any

from .llm.base import LLMResponse
from .llm.providers import get_provider
from .memory import Session
from .prompts import system_prompt
from .tools.definitions import build_registry

MAX_STEPS = 6  # tool calls per user turn


@dataclass
class Turn:
    reply: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    escalated: bool = False


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
                session.add("assistant", reply)
                return Turn(reply=reply, trace=trace,
                            citations=list(dict.fromkeys(session.turn_citations)),
                            escalated=session.escalated)

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
        session.add("assistant", reply)
        return Turn(reply=reply, trace=trace, escalated=True)


def _redact(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the trace readable in the UI without dumping whole payloads."""
    trimmed = json.loads(json.dumps(result, default=str))
    trimmed.pop("prompt", None)
    trimmed.pop("calls", None)   # rendered separately, above the payload
    return trimmed
