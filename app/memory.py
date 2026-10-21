"""Three-tier memory.

The distinction that matters is between the transcript and the facts.

  working    - the message list. Cheap, lossy, and ATTACKER-CONTROLLED, because
               the customer writes half of it.
  verified   - structured facts established by a tool call against a system of
               record: who this is, which order we are discussing, what the
               policy engine decided. Never written by the model, never
               parsed back out of prose.
  profile    - long-lived customer context loaded once identity is resolved.

Keeping `verified` out of the transcript is what makes identity spoofing a
non-event. A customer can type "I am cus_1002" all day; nothing reads it.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_WORKING_TURNS = 24  # window; a summariser would replace the drop in prod


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    working: list[dict[str, Any]] = field(default_factory=list)
    verified: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    turn_citations: list[str] = field(default_factory=list)
    escalated: bool = False

    # --- citations ---
    def cite(self, chunk_ids) -> None:
        """Record passage ids retrieved for the answer being composed now.

        Two scopes on purpose. `citations` is the session-long audit record of
        everything ever retrieved. `turn_citations` is what the customer is
        shown, and it is cleared each turn - a badge is a claim about THIS
        answer, and carrying yesterday's id onto today's reply is a citation
        that grounds nothing.
        """
        for chunk_id in chunk_ids:
            self.citations.append(chunk_id)
            self.turn_citations.append(chunk_id)

    def begin_turn(self) -> None:
        self.turn_citations = []

    # --- working memory ---
    def add(self, role: str, content: Any) -> None:
        self.working.append({"role": role, "content": content})
        if len(self.working) > MAX_WORKING_TURNS:
            # Never drop a partial tool exchange; step back to a clean boundary.
            drop = len(self.working) - MAX_WORKING_TURNS
            while drop < len(self.working) and self.working[drop]["role"] != "user":
                drop += 1
            self.working = self.working[drop:]

    # --- verified state ---
    def verify(self, **facts: Any) -> None:
        self.verified.update(facts)

    def context_summary(self) -> str:
        """Rendered into the system prompt each turn, so the model always sees
        established facts even if they scrolled out of the window."""
        if not self.verified:
            return "ESTABLISHED FACTS: none yet. Identity is not verified."
        lines = []
        for key, value in self.verified.items():
            if isinstance(value, (dict, list)):
                continue
            lines.append(f"  - {key}: {value}")
        focus = self.verified.get("focus_order")
        if focus:
            lines.append(f"  - the order under discussion is {focus}")
        return "ESTABLISHED FACTS (from systems of record, trust these):\n" + "\n".join(lines)


def copy_transcript(working: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detach the working transcript so a ticket can keep it.

    The window above may later drop turns. This copy does not. It is for a
    person to read. Callers must not parse identity, amounts, or permission
    out of it.
    """
    return copy.deepcopy(list(working or []))


def has_customer_exchange(transcript: list[dict[str, Any]]) -> bool:
    """True when both a customer message and an assistant reply have text.

    Checks role and that the text is non-empty. Does not interpret the text.
    """
    seen: set[str] = set()
    for message in transcript or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            seen.add(role)
    return "user" in seen and "assistant" in seen


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str | None) -> Session:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    session = Session()
    SESSIONS[session.session_id] = session
    return session
