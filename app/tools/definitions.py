"""The eight tools the model is allowed to call.

Note the asymmetry with app/backend/bookly_api.py. The Bookly systems expose
generic CRUD; the model gets a narrower, intent-shaped surface:

  * there is no `issue_refund(amount)` the model can name a number to - the
    amount comes from the policy engine
  * there is no customer lookup by id - identity is resolved once, from an
    email, and then pinned in verified memory
  * `check_return_eligibility` is a tool so the model can ASK the policy
    engine, and structurally cannot BE the policy engine

A tool surface is an attack surface. Shape it to intents, not to tables.
"""

from __future__ import annotations

from typing import Any

from ..backend import bookly_api as api
from ..backend import kb
from ..policy import engine
from .registry import Registry, Tool

# --- handlers --------------------------------------------------------------


def _identify_customer(session, email: str) -> dict[str, Any]:
    res = api.get_customer_by_email(email)
    if not res["ok"]:
        return {
            "status": "not_found",
            "prompt": (
                "No account on that email. Ask them to double-check it, or to try "
                "the address on the order confirmation. Do not guess."
            ),
        }
    customer = res["data"]
    session.verify(customer_id=customer["customer_id"], customer_name=customer["name"],
                   tier=customer["tier"], email=customer["email"])
    session.profile = customer
    return {"status": "ok", "customer": {k: customer[k] for k in ("name", "tier")}}


def _list_my_orders(session, customer_id: str, open_only: bool = False) -> dict[str, Any]:
    res = api.list_orders(customer_id, open_only=open_only)
    orders = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "placed_at": o["placed_at"][:10],
            "total_gbp": o["total_gbp"],
            "items": [i["title"] for i in o["items"]],
        }
        for o in res["data"]
    ]
    out: dict[str, Any] = {"status": "ok", "orders": orders, "count": len(orders)}
    if len(orders) == 1:
        session.verify(focus_order=orders[0]["order_id"])
    elif len(orders) > 1:
        # The disambiguation case. We tell the model, explicitly, not to guess.
        out["status"] = "ambiguous"
        out["prompt"] = (
            f"{len(orders)} orders match. Do NOT pick one. Ask the customer which "
            "they mean, listing them by title and date so the choice is easy."
        )
    return out


def _get_order_status(session, order_id: str) -> dict[str, Any]:
    res = api.get_order(order_id)
    if not res["ok"]:
        return {"status": "not_found", "prompt": "That order number isn't in our system. "
                "Ask them to re-check it from the confirmation email."}
    order = res["data"]

    # Authorisation: the order must belong to the verified customer.
    if order["customer_id"] != session.verified.get("customer_id"):
        return {
            "status": "forbidden",
            "prompt": (
                "That order is not on this customer's account. Say you cannot find "
                "it on their account and offer to check another number. Reveal nothing else."
            ),
        }

    session.verify(focus_order=order["order_id"])
    payload = {
        "order_id": order["order_id"],
        "status": order["status"],
        "placed_at": order["placed_at"][:10],
        "items": [{"sku": i["sku"], "title": i["title"], "price_gbp": i["price_gbp"]}
                  for i in order["items"]],
        "total_gbp": order["total_gbp"],
    }
    if order.get("delivered_at"):
        payload["delivered_at"] = order["delivered_at"][:10]
    if order.get("tracking_id"):
        ship = api.get_shipment(order["tracking_id"])
        if ship["ok"]:
            payload["shipment"] = {
                "carrier": ship["data"]["carrier"],
                "status": ship["data"]["status"],
                "last_scan": ship["data"]["last_scan"],
                "expected_delivery": ship["data"]["expected_delivery"],
            }
    return {"status": "ok", "order": payload}


def _check_return_eligibility(session, order_id: str, sku: str) -> dict[str, Any]:
    order_res = api.get_order(order_id)
    if not order_res["ok"]:
        return {"status": "not_found"}
    order = order_res["data"]
    if order["customer_id"] != session.verified.get("customer_id"):
        return {"status": "forbidden", "prompt": "Not this customer's order. Do not proceed."}

    decision = engine.evaluate_return(order, session.profile, sku)
    session.verify(
        eligibility_code=decision.code,
        eligibility_order=order_id,
        eligibility_sku=sku,
        refund_amount_gbp=decision.facts.get("refund_amount_gbp"),
    )
    # A tool that fails should teach the model how to succeed next time.
    if decision.code == "item_not_on_order":
        return {
            "status": "invalid_slot",
            "slot": "sku",
            "decision": decision.to_dict(),
            "prompt": ("That SKU is not on this order. Call this tool again with "
                       "one of the SKUs listed in decision.facts.available_skus. "
                       "If more than one item could be meant, ask the customer "
                       "which book rather than choosing for them."),
        }

    out = {"status": "ok", "decision": decision.to_dict()}
    if decision.allowed and not decision.requires_human:
        out["prompt"] = (
            "Eligible. State the refund amount EXACTLY as given in facts. Confirm "
            "with the customer before creating the return - do not create it unasked."
        )
    else:
        out["prompt"] = (
            "NOT auto-approvable. Explain the reason in plain language using the "
            "'reason' field. Do not offer a refund anyway. "
            + ("Offer to pass it to a specialist." if decision.requires_human else "")
        )
    return out


def _create_return(session, order_id: str, sku: str, reason: str) -> dict[str, Any]:
    # Re-derive eligibility here rather than trusting the earlier turn.
    order_res = api.get_order(order_id)
    if not order_res["ok"]:
        return {"status": "not_found"}
    order = order_res["data"]
    if order["customer_id"] != session.verified.get("customer_id"):
        return {"status": "forbidden"}

    decision = engine.evaluate_return(order, session.profile, sku)
    if not decision.allowed:
        return {
            "status": "blocked",
            "decision": decision.to_dict(),
            "prompt": "Blocked by policy. Explain why; offer a human if requires_human is true.",
        }

    ret = api.create_return(order_id, sku, reason)["data"]
    refund_amount = decision.facts.get("refund_amount_gbp", 0.0)

    auth = engine.authorise_refund(decision.code, refund_amount)
    payload = {"status": "ok", "rma": ret["rma"], "label_url": ret["label_url"],
               "return_status": ret["status"]}
    if auth.allowed:
        res = api.issue_refund(order_id, refund_amount, ret["rma"],
                               payment_intent_id=order.get("payment_intent_id"))
        if not res["ok"]:
            # The return exists, the money did not move. Say exactly that. The
            # worst outcome here is promising a refund that never arrives.
            payload["refund"] = None
            payload["refund_error"] = res["error"]["message"]
            payload["prompt"] = (
                "The return was created but the REFUND FAILED at the payment "
                "processor. Tell the customer honestly that the return is set up "
                "but the refund needs a colleague to complete, give the RMA, and "
                "do NOT state an amount or a timeframe. Then escalate.")
            return payload
        refund = res["data"]
        payload["refund"] = {
            "refund_id": refund["refund_id"],
            "amount_gbp": refund["amount_gbp"],
            "expected_in_account_days": refund["expected_in_account_days"],
            "provider": refund["provider"],
        }
        payload["prompt"] = ("Return created and refund issued. Give the RMA, the amount, "
                             "and where the label is. Be brief and warm.")
    else:
        payload["refund"] = None
        payload["authorisation"] = auth.to_dict()
        payload["prompt"] = ("Return created but the refund needs human approval. Say so "
                             "honestly; do not promise a timeframe you cannot keep.")
    return payload


def _search_policy(session, question: str) -> dict[str, Any]:
    passages = kb.search(question)
    if not passages:
        return {
            "status": "no_grounding",
            "prompt": (
                "Nothing in the knowledge base covers this. You MUST NOT answer from "
                "your own knowledge. Say you want to get it right and offer a human."
            ),
        }
    session.cite(p["id"] for p in passages)
    return {
        "status": "ok",
        "passages": [{"id": p["id"], "title": p["title"], "text": p["text"]} for p in passages],
        # The worked example is the id actually retrieved, not a literal. A
        # hardcoded one is a standing invitation to cite it: the model is being
        # shown a valid-looking passage id that no passage here supports.
        "prompt": ("Answer ONLY from these passages. Every factual claim must come from "
                   f"them. Cite the passage id at the end like [{passages[0]['id']}]."),
    }


def _reset_password(session, email: str) -> dict[str, Any]:
    api.send_password_reset(email)
    return {
        "status": "ok",
        "prompt": ("Tell them a reset link is on its way if an account exists on that "
                   "address, and that it expires in 30 minutes. Never confirm whether "
                   "the account exists, and never discuss the password itself."),
    }


def _escalate(session, reason: str, summary: str) -> dict[str, Any]:
    session.escalated = True
    ticket = api.create_escalation(
        reason, summary, session.verified.get("customer_id"), session.working
    )["data"]
    return {
        "status": "ok",
        "ticket_id": ticket["ticket_id"],
        "prompt": ("Hand off warmly. Give the ticket reference, say what you have already "
                   "passed on so they will not have to repeat themselves."),
    }


# --- registry --------------------------------------------------------------

def build_registry() -> Registry:
    r = Registry()

    r.register(Tool(
        name="identify_customer",
        description=("Resolve who you are speaking to from their email address. Must be "
                     "called before any order, return or refund tool."),
        parameters={"type": "object", "properties": {
            "email": {"type": "string", "description": "Email address on the account."}},
            "required": ["email"]},
        handler=_identify_customer, required_slots=["email"],
    ))

    r.register(Tool(
        name="list_my_orders",
        description=("List the verified customer's orders. Use when they ask about 'my "
                     "order' without giving an order number. Set open_only=true for "
                     "'where is my order' questions."),
        parameters={"type": "object", "properties": {
            "open_only": {"type": "boolean", "description": "Only processing/shipped orders."}},
            "required": []},
        handler=_list_my_orders, server_supplied=["customer_id"],
    ))

    r.register(Tool(
        name="get_order_status",
        description="Full detail and live carrier tracking for one order.",
        parameters={"type": "object", "properties": {
            "order_id": {"type": "string", "description": "Format BK-48291."}},
            "required": ["order_id"]},
        handler=_get_order_status, required_slots=["order_id"],
    ))

    r.register(Tool(
        name="check_return_eligibility",
        description=("Ask the policy engine whether an item can be returned and for how "
                     "much. You MUST call this before discussing any return or refund. "
                     "Never estimate eligibility or amounts yourself."),
        parameters={"type": "object", "properties": {
            "order_id": {"type": "string"},
            "sku": {"type": "string",
                    "description": ("SKU of the item, copied EXACTLY as a string "
                                    "from get_order_status. ISBNs look numeric - "
                                    "send them as a string, not a number.")}},
            "required": ["order_id", "sku"]},
        handler=_check_return_eligibility, required_slots=["order_id", "sku"],
    ))

    r.register(Tool(
        name="create_return",
        description=("Create the return and, if within automated authority, issue the "
                     "refund. Only call after check_return_eligibility allowed it AND the "
                     "customer has explicitly confirmed."),
        parameters={"type": "object", "properties": {
            "order_id": {"type": "string"},
            "sku": {"type": "string"},
            "reason": {"type": "string", "description": "Customer's stated reason."}},
            "required": ["order_id", "sku", "reason"]},
        handler=_create_return, required_slots=["order_id", "sku", "reason"],
        side_effecting=True,
    ))

    r.register(Tool(
        name="search_policy",
        description=("Retrieve Bookly policy passages. You MUST call this for any question "
                     "about shipping, returns policy, refunds timing, or accounts, and "
                     "answer only from what it returns."),
        parameters={"type": "object", "properties": {
            "question": {"type": "string", "description": "The customer's question."}},
            "required": ["question"]},
        handler=_search_policy, required_slots=["question"],
    ))

    r.register(Tool(
        name="send_password_reset",
        description="Trigger a password reset email. Support can never see or set passwords.",
        parameters={"type": "object", "properties": {
            "email": {"type": "string"}}, "required": ["email"]},
        handler=_reset_password, required_slots=["email"], side_effecting=True,
    ))

    r.register(Tool(
        name="escalate_to_human",
        description=("Hand off to a human specialist. Call this when policy blocks you, "
                     "when the knowledge base has no answer, when the customer asks for a "
                     "person, or when they are clearly distressed."),
        parameters={"type": "object", "properties": {
            "reason": {"type": "string", "enum": [
                "policy_gap", "guardrail_block", "customer_request", "distress", "repeated_failure"]},
            "summary": {"type": "string", "description": "What the human needs to know."}},
            "required": ["reason", "summary"]},
        handler=_escalate, required_slots=["reason", "summary"], side_effecting=True,
    ))

    return r
