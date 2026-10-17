"""Tool registry with pre-execution slot validation.

Why this exists: "ask the customer for their order number before looking it up"
is a rule you can put in a prompt and the model will follow it most of the time.
Most of the time is not a spec.

So required slots are declared on the tool, and the registry refuses to execute
a tool whose slots are missing or malformed. It returns a NEED_SLOT envelope
instead. The clarifying question is therefore a property of the architecture,
and the model's job is only to phrase it well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..backend import wire

ORDER_RE = re.compile(r"^BK-\d{5}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "order_id": lambda v: bool(ORDER_RE.match(str(v).strip().upper())),
    "email": lambda v: bool(EMAIL_RE.match(str(v).strip())),
}

# Human-readable prompts the model is told to use when a slot is missing.
SLOT_PROMPTS = {
    "order_id": "the order number (it looks like BK-48291, on your confirmation email)",
    "email": "the email address on the order",
    "sku": "which item from the order",
    "reason": "the reason for the return",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema, sent to the LLM verbatim
    handler: Callable[..., dict]
    required_slots: list[str] = field(default_factory=list)
    side_effecting: bool = False
    # Slots the model may never supply itself; they are injected from verified
    # session state. Stops "I'm customer cus_1002" style identity spoofing.
    server_supplied: list[str] = field(default_factory=list)
    # Optional per-tool wording when a slot is missing or malformed. Falls
    # back to SLOT_PROMPTS. Cancel must not reuse the returns prompt that
    # sends the customer hunting through a confirmation email.
    slot_prompts: dict[str, str] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any], session) -> dict[str, Any]:
        """Validate, then run. Never raises into the agent loop."""
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error", "error": f"Unknown tool '{name}'."}

        args = dict(args or {})

        # 1. Strip anything the model is not allowed to assert, then inject the
        #    trusted value from session state.
        for slot in tool.server_supplied:
            args.pop(slot, None)
            trusted = session.verified.get(slot)
            if trusted is None:
                return {
                    "status": "need_slot",
                    "missing": ["email"],
                    "prompt": (
                        "Identity is not established yet. Ask for "
                        f"{SLOT_PROMPTS['email']} before continuing."
                    ),
                }
            args[slot] = trusted

        # 2. Required-slot check -> forced clarifying question.
        missing = [
            s
            for s in tool.required_slots
            if args.get(s) in (None, "", "unknown", "null")
        ]
        if missing:
            return {
                "status": "need_slot",
                "missing": missing,
                "prompt": (
                    "Do not call this tool yet. Ask the customer for "
                    + " and ".join(self._slot_prompt(tool, s) for s in missing)
                    + ". Ask for it naturally, in one short question."
                ),
            }

        # 3. Coerce to the declared JSON Schema types.
        #    Models do not reliably honour `"type": "string"`. An ISBN looks
        #    numeric, so `sku` arrives as 9780099540946 rather than the string,
        #    and every downstream comparison silently fails. Coercing here is
        #    one place; chasing it through the policy engine is many.
        props = (tool.parameters or {}).get("properties", {})
        for slot, value in list(args.items()):
            declared = props.get(slot, {}).get("type")
            if declared == "string" and isinstance(value, (int, float)):
                args[slot] = format(value, ".0f") if isinstance(value, float) else str(value)
            elif declared == "boolean" and isinstance(value, str):
                args[slot] = value.strip().lower() in ("true", "yes", "1")

        # 4. Format validation -> repair question rather than a 404.
        #    A blank optional slot is "not supplied", not "supplied wrong".
        #    Validating it forces the order-number prompt on a cancel the
        #    customer has not named an order for.
        for slot, value in list(args.items()):
            if slot not in tool.required_slots and value in (None, "", "unknown", "null"):
                args.pop(slot, None)
                continue
            validator = VALIDATORS.get(slot)
            if validator and not validator(value):
                return {
                    "status": "invalid_slot",
                    "slot": slot,
                    "prompt": (
                        f"'{value}' is not a valid {slot}. Ask the customer to "
                        f"confirm {self._slot_prompt(tool, slot)}."
                    ),
                }

        # Everything this tool sends to a system is captured and attached to the
        # result, so the trace shows the call chain rather than just the outcome.
        wire.begin()
        try:
            result = tool.handler(session=session, **args)
        except Exception as exc:  # a broken tool must degrade, not crash the chat
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "calls": wire.drain(),
                "prompt": "The system call failed. Apologise briefly and offer a human.",
            }

        result.setdefault("status", "ok")
        result["calls"] = wire.drain()
        session.tool_calls.append({"tool": name, "args": args, "status": result["status"]})
        return result

    @staticmethod
    def _slot_prompt(tool: Tool, slot: str) -> str:
        return tool.slot_prompts.get(slot) or SLOT_PROMPTS.get(slot, slot)
