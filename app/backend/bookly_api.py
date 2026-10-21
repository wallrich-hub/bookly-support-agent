"""Mocked Bookly systems of record.

This module is the seam where a real deployment would hold HTTP clients for the
OMS, the payment processor and the ticketing system. Every function here is
written as if it were a network call with a typed response envelope, so that
swapping the body for `httpx.get(...)` is the only change required.

Design note: these are the *system* APIs. They are deliberately NOT the same
surface as the agent's tools. See app/tools/definitions.py for the narrower,
intent-shaped tool layer that the model is actually allowed to call.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone  # noqa: F401  (audit timestamps)
from typing import Any

from ..memory import copy_transcript
from . import data
from .oms import OmsUnavailable, get_oms
from .payments import PaymentError, get_gateway  # noqa: F401


def _audit(action: str, payload: dict[str, Any], result: str) -> None:
    data.AUDIT_LOG.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "payload": payload,
            "result": result,
        }
    )


def _ok(payload: Any) -> dict[str, Any]:
    return {"ok": True, "data": payload}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


# --- Identity --------------------------------------------------------------


def get_customer_by_email(email: str) -> dict[str, Any]:
    """GET /v1/customers?email="""
    customer = get_oms().get_customer_by_email(email)
    if customer is None:
        return _err("customer_not_found", f"No Bookly account found for {email}.")
    return _ok(customer)


# --- Orders ----------------------------------------------------------------


def list_orders(customer_id: str, open_only: bool = False) -> dict[str, Any]:
    """GET /v1/orders?customer_id=&open_only="""
    rows = get_oms().list_orders(customer_id)
    if open_only:
        rows = [o for o in rows if o["status"] in ("processing", "shipped")]
    return _ok(rows)


def get_order(order_id: str) -> dict[str, Any]:
    """GET /v1/orders/{order_id}"""
    order = get_oms().get_order(order_id)
    if not order:
        return _err("order_not_found", f"No order {order_id}.")
    return _ok(order)


def cancel_order(order_id: str) -> dict[str, Any]:
    """POST /v1/orders/{order_id}/cancel  (idempotent on order_id)

    Status only. The refund is a separate call, with its own idempotency key,
    and this function never makes it. Replaying a cancel returns the cancelled
    order and does not write a second record.
    """
    try:
        current = get_oms().get_order(order_id)
    except OmsUnavailable as exc:
        _audit("cancel_order", {"order_id": order_id}, f"failed: {exc}")
        return _err("oms_unavailable", str(exc))
    if not current:
        return _err("order_not_found", f"No order {order_id}.")
    if current.get("status") == "cancelled":
        _audit("cancel_order", {"order_id": order_id}, "replayed")
        return _ok({**current, "replayed": True})
    try:
        cancelled = get_oms().cancel_order(order_id)
    except OmsUnavailable as exc:
        _audit("cancel_order", {"order_id": order_id}, f"failed: {exc}")
        return _err("oms_unavailable", str(exc))
    if not cancelled:
        _audit("cancel_order", {"order_id": order_id}, "failed")
        return _err("cancel_failed", f"Could not cancel {order_id}.")
    _audit("cancel_order", {"order_id": order_id}, "cancelled")
    return _ok(cancelled)


def get_shipment(tracking_id: str) -> dict[str, Any]:
    """GET /v1/shipments/{tracking_id}"""
    ship = get_oms().get_shipment(tracking_id)
    if not ship:
        return _err("shipment_not_found", f"No shipment {tracking_id}.")
    return _ok(ship)


# --- Returns and refunds (side-effecting, idempotent) ----------------------


def _idem(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def create_return(order_id: str, sku: str, reason: str) -> dict[str, Any]:
    """POST /v1/returns  (idempotent on order_id+sku)"""
    key = _idem(order_id, sku)
    if key in data.RETURNS:
        _audit("create_return", {"order_id": order_id, "sku": sku}, "replayed")
        return _ok({**data.RETURNS[key], "replayed": True})
    rma = f"RMA-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "rma": rma,
        "order_id": order_id,
        "sku": sku,
        "reason": reason,
        "status": "awaiting_shipment",
        "label_url": f"https://bookly.example/labels/{rma}.pdf",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data.RETURNS[key] = record
    _audit("create_return", {"order_id": order_id, "sku": sku, "reason": reason}, rma)
    return _ok(record)


def issue_refund(order_id: str, amount_gbp: float, rma: str,
                 payment_intent_id: str | None = None) -> dict[str, Any]:
    """Move the money, through whichever gateway is configured.

    This is the only writer of data.REFUNDS, and only after the gateway
    succeeds. Chat prose never creates a row, and a failed call is not retried.

    Two layers of idempotency, deliberately. The local check short-circuits a
    replay without a network call; the gateway is ALSO given the RMA as an
    idempotency key, so a replay that gets past us (different process, restarted
    server) still cannot double-refund. Belt and braces on the irreversible step.
    """
    if rma in data.REFUNDS:
        _audit("issue_refund", {"rma": rma}, "replayed")
        return _ok({**data.REFUNDS[rma], "replayed": True})

    try:
        result = get_gateway().refund(
            order_id=order_id, amount_gbp=amount_gbp, rma=rma,
            payment_intent_id=payment_intent_id,
        )
    except PaymentError as exc:
        # Never fabricate a refund record for money that did not move.
        _audit("issue_refund", {"order_id": order_id, "rma": rma}, f"failed: {exc}")
        return _err("refund_failed", str(exc))

    record = {
        **result,
        "order_id": order_id,
        "rma": rma,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data.REFUNDS[rma] = record
    _audit("issue_refund",
           {"order_id": order_id, "amount_gbp": amount_gbp, "provider": result["provider"]},
           record["refund_id"])
    return _ok(record)


# --- Account ---------------------------------------------------------------


def send_password_reset(email: str) -> dict[str, Any]:
    """POST /v1/auth/password-reset

    Deliberately returns the same shape whether or not the account exists, so
    the agent cannot be used as an account-enumeration oracle.
    """
    data.PASSWORD_RESETS.append(
        {"email": email, "at": datetime.now(timezone.utc).isoformat()}
    )
    _audit("send_password_reset", {"email": email}, "sent_if_exists")
    return _ok({"sent": True, "expires_minutes": 30})


# --- Human handoff ---------------------------------------------------------


def create_escalation(
    reason: str, summary: str, customer_id: str | None, transcript: list[dict]
) -> dict[str, Any]:
    """POST /v1/escalations -> ticketing system

    The transcript is stored so a specialist can read the turns. It is
    customer-written and untrusted: this function does not read identity,
    amounts, or permission out of it.
    """
    ticket = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    stored = copy_transcript(transcript)
    data.ESCALATIONS[ticket] = {
        "ticket_id": ticket,
        "reason": reason,
        "summary": summary,
        "customer_id": customer_id,
        "priority": "high" if reason in ("policy_gap", "guardrail_block") else "normal",
        "transcript_turns": len(stored),
        "transcript": stored,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _audit("create_escalation", {"reason": reason}, ticket)
    return _ok(data.ESCALATIONS[ticket])
