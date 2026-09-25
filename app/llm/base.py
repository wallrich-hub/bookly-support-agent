"""Provider-agnostic LLM interface.

The orchestrator is written against this, not against any vendor SDK. Two
reasons, both of which are enterprise-deal reasons rather than engineering
taste: a customer's model choice is often a procurement or data-residency
decision rather than a technical one, and the ability to route cheap intents to
a small model is where the unit economics of support automation actually live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalised across vendors: either text, or tool calls, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        ...
