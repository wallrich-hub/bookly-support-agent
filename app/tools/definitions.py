"""The ten tools the model is allowed to call.

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

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def format_uk_date(value: str | None) -> str | None:
    """Speak an ISO date as '20 October 2026'.

    The tool value stays the source of the calendar day. The model is not
    asked to convert it: a wrong conversion is how a delivered parcel was
    announced as due on the carrier's estimate.
    """
    text = (value or "").strip()
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        year, month, day = (int(text[:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{day} {_MONTHS[month - 1]} {year}"


def delivery_line(status: str, delivered_at: str | None, shipment: dict | None) -> str | None:
    """The sentence a status reply may quote. None when there is no shipment.

    A delivered parcel speaks delivered_at. An in-transit parcel speaks the
    carrier expected_delivery as an estimate. Neither sentence says the parcel
    is "due" on the other date.
    """
    if not shipment:
        return None
    ship_status = str(shipment.get("status") or "")
    delivered = status == "delivered" or ship_status == "delivered" or bool(delivered_at)
    if delivered:
        spoken = format_uk_date(delivered_at)
        if spoken:
            return f"It was delivered on {spoken}."
        return "It has been delivered."
    spoken = format_uk_date(str(shipment.get("expected_delivery") or ""))
    if spoken:
        return f"The carrier estimates delivery on {spoken}."
    return None


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
    line = delivery_line(
        order["status"], payload.get("delivered_at"), payload.get("shipment"),
    )
    out: dict[str, Any] = {"status": "ok", "order": payload}
    if line:
        payload["delivery_line"] = line
        out["prompt"] = (
            "Tell them the status. Say delivery_line exactly as written. "
            "Do not convert the date, and do not say the parcel is due on any other date."
        )
    return out


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


def _cancel_confirmed(session, order_id: str) -> bool:
    """True only when the customer's latest message is the Cancel click.

    The model can call this tool whenever it likes. The write waits for the
    message the server put on the action, which a click sends back through chat.
    """
    expected = engine.cancel_confirmation(order_id)
    for message in reversed(session.working):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        return content.strip() == expected
    return False


def _refund_view(refund: dict) -> dict[str, Any]:
    return {
        "refund_id": refund["refund_id"],
        "amount_gbp": refund["amount_gbp"],
        "expected_in_account_days": refund["expected_in_account_days"],
        "provider": refund["provider"],
    }


def _forbid_order() -> dict[str, Any]:
    return {
        "status": "forbidden",
        "prompt": (
            "That order is not on this customer's account. Say you cannot find "
            "it on their account and offer to check another number. Reveal nothing else."
        ),
    }


def _cancel_order(session, customer_id: str, order_id: str | None = None, **_extra) -> dict[str, Any]:
    """Cancel one unshipped order and refund its total.

    `_extra` absorbs an amount or a customer id if a model invents one.
    Neither is read. Identity is `customer_id` from the session; the amount
    is re-derived from the order at write time.
    """
    named = (order_id or "").strip()
    if named.lower() in ("", "unknown", "null", "none"):
        named = ""

    if named:
        got = api.get_order(named)
        if not got["ok"]:
            return {
                "status": "not_found",
                "prompt": "That order number isn't in our system. Ask them to re-check it.",
            }
        order = got["data"]
        if order["customer_id"] != customer_id:
            return _forbid_order()
    else:
        processing = [
            row for row in api.list_orders(customer_id)["data"]
            if row.get("status") == "processing"
        ]
        if len(processing) == 1:
            order = processing[0]
        elif not processing:
            decision = engine.Decision(
                allowed=False,
                code="nothing_to_cancel",
                reason="There isn't an unshipped order on this account.",
            )
            return {
                "status": "blocked",
                "decision": decision.to_dict(),
                "prompt": (
                    "Explain that there is nothing to cancel. "
                    "Do not say an order can be cancelled."
                ),
            }
        else:
            lines = "; ".join(
                f"{row['items'][0]['title']} placed {row['placed_at'][:10]}"
                for row in processing
            )
            return {
                "status": "ambiguous",
                "prompt": (
                    "More than one unshipped order. Ask which they mean by title and date: "
                    f"{lines}. Do not ask them to copy an order number from a confirmation email."
                ),
            }

    # Write-time re-read. The copy from a moment ago is not authority to move money.
    fresh = api.get_order(order["order_id"])
    if not fresh["ok"]:
        return {"status": "not_found", "prompt": "That order number isn't in our system."}
    order = fresh["data"]
    if order["customer_id"] != customer_id:
        return _forbid_order()

    decision = engine.evaluate_cancel(order, customer_id)
    session.verify(focus_order=order["order_id"], cancel_code=decision.code)

    if not decision.allowed or not _cancel_confirmed(session, order["order_id"]):
        out: dict[str, Any] = {
            "status": "ok" if decision.allowed else "blocked",
            "decision": decision.to_dict(),
            "order_id": order["order_id"],
        }
        if decision.allowed:
            out["prompt"] = (
                "Allowed. Tell them briefly that it can be cancelled. The Cancel "
                "action is shown separately. Do not cancel it yourself and do not "
                "ask them to find an order number."
            )
        else:
            out["prompt"] = (
                "Not cancellable. Explain the reason in plain language. "
                "Do not say it can be cancelled."
            )
        return out

    amount = decision.facts["refund_amount_gbp"]
    # Stable key: a second click replays this refund instead of creating one.
    # issue_refund does not retry the POST; neither does this caller.
    res = api.issue_refund(
        order["order_id"],
        amount,
        f"CANCEL-{order['order_id']}",
        payment_intent_id=order.get("payment_intent_id"),
    )
    if not res["ok"]:
        return {
            "status": "blocked",
            "refund": None,
            "refund_error": res["error"]["message"],
            "order_id": order["order_id"],
            "prompt": (
                "The refund failed and the order was NOT cancelled. Tell them a "
                "colleague needs to finish it. Do not state an amount, a timeframe, "
                "or that it can be cancelled."
            ),
        }

    cancelled = api.cancel_order(order["order_id"])
    if not cancelled["ok"]:
        # The money moved once, under the idempotency key. A later confirmed
        # click calls issue_refund again and gets the same refund back.
        return {
            "status": "blocked",
            "refund": _refund_view(res["data"]),
            "order_id": order["order_id"],
            "prompt": (
                "The refund was issued but the order status did not update. "
                "Do not take the money again. Say the refund is on its way and a "
                "colleague will confirm the cancellation. Do not state a second amount."
            ),
        }

    session.verify(cancel_code="cancelled")
    return {
        "status": "ok",
        "order_id": order["order_id"],
        "refund": _refund_view(res["data"]),
        "prompt": (
            "Cancelled and refunded. Give the amount and the timing. Be brief. "
            "Do not say it can be cancelled — it already is."
        ),
    }


def _latest_user_text(session) -> str:
    for message in reversed(session.working):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return content if isinstance(content, str) else ""
    return ""


def _record_reply_feedback(session, **_extra) -> dict[str, Any]:
    """Record a thumbs click. The rating and the reply come from the click.

    `_extra` absorbs a rating, a reply, or a customer id if a model invents
    one. None of those are read. Identity is the session; the reply text is
    the sentence the server stored for that reply id.
    """
    target = engine.feedback_target(session, _latest_user_text(session))
    if target is None:
        decision = engine.evaluate_feedback("", "", False)
        return {
            "status": "blocked",
            "decision": decision.to_dict(),
            "prompt": "There is no thumbs click to record. Do not invent a rating.",
        }
    rating, reply = target
    decision = engine.evaluate_feedback(rating, reply["text"], True)
    if not decision.allowed:
        return {
            "status": "blocked",
            "decision": decision.to_dict(),
            "prompt": "There is no thumbs click to record. Do not invent a rating.",
        }
    saved = api.record_reply_feedback(
        session.session_id, reply["id"], reply["text"], rating,
    )["data"]
    return {
        "status": "ok",
        "decision": decision.to_dict(),
        "feedback": {"rating": saved["rating"], "reply_id": saved["reply_id"]},
        "acknowledgement": engine.feedback_acknowledgement(rating),
        "prompt": (
            "The rating is already recorded. The acknowledgement is fixed. "
            "Do not add to it, and do not refund, cancel, or take a payment."
        ),
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
                     "called before any order, return, cancellation or refund tool."),
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
        name="cancel_order",
        description=(
            "Cancel the verified customer's unshipped order and refund the order "
            "total when policy allows. You cannot name an amount or a customer. "
            "Omit order_id unless they named one. When the decision is allowed, "
            "a Cancel action is shown for them; do not cancel until they use it, "
            "and do not say an order can be cancelled unless the decision allows it."
        ),
        parameters={"type": "object", "properties": {
            "order_id": {
                "type": "string",
                "description": "Only if the customer named an order. Otherwise omit.",
            },
        }, "required": []},
        handler=_cancel_order,
        required_slots=[],
        server_supplied=["customer_id"],
        side_effecting=True,
        slot_prompts={
            "order_id": "which order they mean, by the title if there is more than one",
        },
    ))

    r.register(Tool(
        name="search_policy",
        description=("Retrieve Bookly policy passages. You MUST call this for any question "
                     "about shipping, returns policy, cancellation, refunds timing, or "
                     "accounts, and answer only from what it returns."),
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
        name="record_reply_feedback",
        description=(
            "Record a thumbs up or thumbs down the customer has already clicked. "
            "The rating and the reply are taken from that click. You cannot name "
            "a rating, a reply, or a customer. Do not call this for a complaint "
            "typed in prose; calling it then records nothing."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_record_reply_feedback,
        required_slots=[],
        side_effecting=True,
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
