"""Concrete providers: Anthropic, OpenAI, and a deterministic mock.

The mock is NOT pretending to be an LLM. It is a small rule-based planner that
speaks the exact same wire format, so the repo runs and the tests assert
end-to-end with no API key and no network. It exists to prove the claim the
architecture makes: if the orchestration, policy and tool layers are doing the
real work, a much weaker reasoner still produces a safe, correct conversation.
Swap in Claude or GPT and the conversation gets more fluent - not more correct.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from .base import LLMResponse, ToolCall

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ORDER_RE = re.compile(r"\bBK-?\s?(\d{5})\b", re.I)
INJECTION_RE = re.compile(
    r"ignore (all |your |previous |prior )*(instructions|rules|prompt)"
    r"|developer mode|you are now|system prompt|disregard (all|your|previous)",
    re.I,
)


# =========================================================================
# Anthropic
# =========================================================================
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        import anthropic  # imported lazily so the repo runs without the SDK

        # base_url is overridable so the provider can be pointed at a gateway,
        # a proxy, or a stub in tests. Never guessed - explicit or default.
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **({"base_url": os.environ["ANTHROPIC_BASE_URL"]}
               if os.environ.get("ANTHROPIC_BASE_URL") else {}),
        )
        self.model = model

    def complete(self, system, messages, tools) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=_to_anthropic(messages),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMResponse(text=text.strip(), tool_calls=calls, raw=resp)


def _to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                content.append({"type": "tool_use", "id": tc["id"],
                                "name": tc["name"], "input": tc["arguments"]})
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": m["tool_call_id"],
                "content": json.dumps(m["content"])}]})
    return out


# =========================================================================
# OpenAI
# =========================================================================
class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o") -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, system, messages, tools) -> LLMResponse:
        oai_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}} for t in tools]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + _to_openai(messages),
            tools=oai_tools,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name,
                     arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls, raw=resp)


def _to_openai(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"])}}
                    for tc in m["tool_calls"]]
            out.append(entry)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": json.dumps(m["content"])})
    return out


# =========================================================================
# Deterministic mock planner
# =========================================================================
def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=uuid.uuid4().hex[:8], name=name, arguments=args)])


def _say(text: str) -> LLMResponse:
    return LLMResponse(text=text)


class MockProvider:
    """Rule-based planner. Same interface, no network, fully deterministic.

    This is NOT an LLM and does not pretend to be one. It exists so the repo
    runs and the suite passes with no credentials, and as a control: if the
    architecture is carrying the safety and correctness load, a deliberately
    weak reasoner still cannot refund the wrong amount or answer uncited. It
    cannot generalise, handle unanticipated phrasing, or show any judgement -
    which is exactly the half of the job a real model is for.
    """

    name = "mock (no LLM)"

    RETURN_WORDS = ("return", "refund", "send back", "money back", "damaged", "faulty")
    STATUS_WORDS = ("where", "order status", "track", "arrived", "delivery of my", "my order")
    POLICY_WORDS = ("policy", "how long", "shipping", "postage", "cost", "delivered",
                    "free delivery", "next day", "lost", "missing")
    HUMAN_WORDS = ("human", "person", "agent", "manager", "speak to someone")
    PASSWORD_WORDS = ("password", "log in", "login", "sign in", "locked out")

    # Asking what the rules ARE is a different intent from asking us to act on
    # one order, even though both use the words "return" and "refund". Without
    # this split every policy question containing "return", "refund", "where" or
    # "lost" was captured by the transactional branches below, so search_policy
    # was only ever reached by shipping questions - and the only citation the
    # agent could physically emit was KB-SHIP-01, whatever was asked.
    GENERAL_ASK = ("your policy", "your returns", "your refund", "what is your",
                   "what's your", "how long", "how do i", "how does", "can i",
                   "do you", "does bookly", "what happens", "when do", "when will",
                   "is delivery", "are ebooks", "policy")
    SPECIFIC_REF = ("my order", "my parcel", "my package", "my delivery", "my item",
                    "this order", "that order", "i ordered", "i bought", "my book",
                    "where is it", "where's it", "track")

    def complete(self, system, messages, tools) -> LLMResponse:
        last = messages[-1] if messages else {}

        if last.get("role") == "tool":
            return self._after_tool(last, messages)

        text = (last.get("content") or "").lower()
        history = " ".join(
            (m.get("content") or "") for m in messages if m.get("role") == "user"
        ).lower()
        verified_email = "email:" in system

        # Injection is not a security control - the policy plane is. But naming
        # it is better CX than silently routing an attacker into a happy path.
        if INJECTION_RE.search(text):
            return _say("I'll help with the real question, but I can't change how I work "
                        "or move money outside our returns process. If you'd like to look "
                        "at a refund, tell me the order and I'll check it properly.")

        # Confirming a return that the policy engine already cleared.
        if re.match(r"^\s*(yes|yep|yeah|please do|go ahead|ok|okay|sure|do it)\b",
                    text) and "eligibility_code: eligible_auto" in system:
            oid = re.search(r"eligibility_order: (\S+)", system).group(1)
            sku = re.search(r"eligibility_sku: (\S+)", system).group(1)
            reason = ("damaged on arrival" if "damag" in history
                      else "not as expected" if "expect" in history else "customer changed mind")
            return _call("create_return", order_id=oid, sku=sku, reason=reason)

        email = EMAIL_RE.search(last.get("content") or "")

        # A general question about the rules, naming no particular order, is a
        # retrieval job. Checked BEFORE the transactional branches so that
        # "what is your returns policy?" reaches the returns chunk instead of
        # being swallowed by the word "return".
        general = (any(w in text for w in self.GENERAL_ASK) or text.endswith("?"))
        specific = (any(w in text for w in self.SPECIFIC_REF)
                    or ORDER_RE.search(last.get("content") or ""))
        if general and not specific:
            return _call("search_policy", question=last.get("content") or history)

        if any(w in text for w in self.HUMAN_WORDS):
            return _call("escalate_to_human", reason="customer_request",
                         summary="Customer asked to speak to a person.")

        # Password intent is checked against the whole conversation, before the
        # email shortcut, so supplying the address does not silently re-route
        # a locked-out customer into order tracking.
        if any(w in history for w in self.PASSWORD_WORDS):
            if not email and not verified_email:
                return _say("Happy to help you get back in. What's the email address on "
                            "your Bookly account?")
            addr = (email.group(0) if email
                    else re.search(r"email: (\S+)", system).group(1))
            return _call("send_password_reset", email=addr)

        # An email in the message is otherwise identity resolution.
        if email and not verified_email:
            return _call("identify_customer", email=email.group(0))

        wants_return = any(w in text for w in self.RETURN_WORDS)
        wants_status = any(w in text for w in self.STATUS_WORDS)

        # An order number in the message pins the order.
        order = ORDER_RE.search(last.get("content") or "")
        if order and verified_email:
            oid = f"BK-{order.group(1)}"
            return _call("get_order_status", order_id=oid)

        if wants_return or wants_status:
            if not verified_email:
                return _say("Of course — I can look that up. What's the email address on "
                            "the order?")
            if wants_status and not wants_return:
                return _call("list_my_orders", open_only=True)
            return _call("list_my_orders", open_only=False)

        if any(w in text for w in self.POLICY_WORDS) or text.endswith("?"):
            return _call("search_policy", question=last.get("content") or history)

        return _say("I can help with order tracking, returns and refunds, or questions "
                    "about delivery and your account. Which of those is it?")

    # --- reactions to tool results ---------------------------------------
    def _after_tool(self, last: dict, messages: list[dict]) -> LLMResponse:
        name = last.get("name")
        res = last.get("content") or {}
        status = res.get("status")
        # Intent persists across turns: a customer who said "return" three turns
        # ago is still returning something. A real LLM gets this from the
        # transcript; the mock has to be told to look at all of it.
        user_text = " ".join(
            (m.get("content") or "") for m in messages if m.get("role") == "user"
        ).lower()

        if status in ("need_slot", "invalid_slot"):
            return _say(res.get("prompt", "Could you give me a little more detail?"))

        if status == "error":
            return _say("Sorry — that lookup failed on our side. Let me get a colleague "
                        "to pick this up for you.")

        if name == "identify_customer":
            if status == "not_found":
                return _say("I can't find an account on that address. Could you check the "
                            "email on your order confirmation and try again?")
            nm = res["customer"]["name"].split()[0]
            wants_return = any(w in user_text for w in self.RETURN_WORDS)
            return _call("list_my_orders", open_only=not wants_return)

        if name == "list_my_orders":
            orders = res.get("orders", [])
            if not orders:
                return _say("I can't see any orders on the account. Is it possible you "
                            "used a different email at checkout?")
            if status == "ambiguous":
                lines = "\n".join(
                    f"• {o['order_id']} — {', '.join(o['items'])} ({o['status']}, "
                    f"placed {o['placed_at']})" for o in orders[:4])
                return _say("I can see a few orders on your account — which one do you "
                            f"mean?\n\n{lines}")
            return _call("get_order_status", order_id=orders[0]["order_id"])

        if name == "get_order_status":
            if status != "ok":
                return _say(res.get("prompt", "I couldn't find that order."))
            o = res["order"]
            if any(w in user_text for w in self.RETURN_WORDS):
                return _call("check_return_eligibility",
                             order_id=o["order_id"], sku=o["items"][0]["sku"])
            ship = o.get("shipment")
            if ship:
                return _say(
                    f"{o['order_id']} is {o['status'].replace('_',' ')} with {ship['carrier']}. "
                    f"Latest scan: {ship['last_scan']}. It's due {ship['expected_delivery']}.")
            return _say(f"{o['order_id']} is still {o['status']} in our Chelmsford warehouse — "
                        "you'll get a tracking email the moment it's dispatched.")

        if name == "check_return_eligibility":
            d = res.get("decision", {})
            if d.get("allowed") and not d.get("requires_human"):
                amt = d["facts"]["refund_amount_gbp"]
                title = d["facts"].get("title", "the item")
                return _say(f"Good news — {title} is inside your return window, so I can "
                            f"refund £{amt:.2f} in full. Shall I set the return up and send "
                            "you a prepaid label?")
            if d.get("requires_human"):
                return _call("escalate_to_human", reason="guardrail_block",
                             summary=f"Return needs human review: {d.get('reason')}")
            return _say(d.get("reason", "That one isn't eligible for return, I'm afraid."))

        if name == "create_return":
            if status != "ok":
                return _say(res.get("decision", {}).get("reason", "I couldn't set that up."))
            refund = res.get("refund")
            if res.get("refund_error"):
                # Return created, money did not move. Never quote an amount or a
                # date we cannot stand behind - hand it to someone who can.
                return _call("escalate_to_human", reason="repeated_failure",
                             summary=(f"Return {res['rma']} created but the refund "
                                      f"failed at the processor: {res['refund_error']}"))
            if refund:
                return _say(
                    f"Done. Your return reference is {res['rma']} and I've issued a "
                    f"£{refund['amount_gbp']:.2f} refund — it'll be back on your card within "
                    f"{refund['expected_in_account_days']} working days. Your prepaid label "
                    f"is here: {res['label_url']}")
            return _say(f"Return {res['rma']} is created and your label is at "
                        f"{res['label_url']}. The refund needs a quick sign-off from a "
                        "colleague, who'll confirm by email today.")

        if name == "search_policy":
            if status == "no_grounding":
                return _call("escalate_to_human", reason="policy_gap",
                             summary="No knowledge base coverage for the question asked.")
            p = res["passages"][0]
            return _say(f"{p['text']}\n\n[{p['id']}]")

        if name == "send_password_reset":
            return _say("I've sent a reset link to that address — if there's a Bookly "
                        "account on it, the email will be with you shortly. The link is "
                        "valid for 30 minutes.")

        if name == "escalate_to_human":
            # Never hand off without saying why. Look back for the decision that
            # caused it and lead with the plain-language reason.
            preface = ""
            for m in reversed(messages):
                if m.get("role") != "tool":
                    continue
                # A refund that failed at the processor outranks anything the
                # eligibility check said - it was eligible, that is the point.
                if m.get("name") == "create_return" and (m.get("content") or {}).get("refund_error"):
                    rma = (m.get("content") or {}).get("rma", "")
                    preface = (f"Your return is set up{f' (ref {rma})' if rma else ''}, but I "
                               "wasn't able to put the refund through myself. ")
                    break
                if m.get("name") == "check_return_eligibility":
                    d = (m.get("content") or {}).get("decision", {})
                    # Only quote the eligibility reason when it is why we stopped.
                    if d.get("reason") and not d.get("allowed"):
                        r = d["reason"]
                        preface = f"I can't complete that one automatically — {r[0].lower()}{r[1:]} "
                    break
                if m.get("name") == "search_policy":
                    preface = ("That's outside what I can confirm from our help centre, and "
                               "I'd rather be right than quick. ")
                    break
            return _say(f"{preface}I've passed this to a specialist — your reference is "
                        f"{res['ticket_id']}. I've included everything from our chat, so "
                        "you won't need to repeat yourself. They'll be in touch shortly.")

        return _say("Let me get a colleague to pick this up for you.")


# =========================================================================
def resolve_provider_name(explicit: str | None = None) -> str:
    """Explicit choice wins; otherwise use whichever key is present.

    Defaulting to the mock when a real key is sitting in the environment is the
    wrong surprise: someone who has configured a model expects the model. The
    mock is the fallback for a machine with no credentials, not the preference.
    """
    name = explicit or os.environ.get("LLM_PROVIDER")
    if name:
        return name.lower()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def get_provider(name: str | None = None):
    name = resolve_provider_name(name)
    if name == "anthropic":
        return AnthropicProvider(os.environ.get("LLM_MODEL", "claude-sonnet-4-5"))
    if name == "openai":
        return OpenAIProvider(os.environ.get("LLM_MODEL", "gpt-4o"))
    return MockProvider()
