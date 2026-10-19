"""Payment gateway tests — offline, against a stubbed Stripe.

What matters about a refund integration is not the happy path. It is:
  * the amount is exactly right, in minor units, with no float drift
  * a replay does not move money twice
  * a failure is never reported to the customer as success
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.backend import data
from app.backend.money import from_minor_units, refund_for_total, to_minor_units
from app.backend.payments import (LedgerGateway, PaymentError, StripeGateway,
                                  set_gateway)
from app.memory import Session
from app.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def clean():
    yield
    set_gateway(None)
    data.RETURNS.clear(); data.REFUNDS.clear(); data.CANCELLATIONS.clear()
    data.ESCALATIONS.clear(); data.AUDIT_LOG.clear()


def stub_stripe(handler, payment_map=None) -> StripeGateway:
    gw = StripeGateway.__new__(StripeGateway)
    gw._key = "sk_test_stub"
    gw._client = httpx.Client(
        base_url="https://api.stripe.com/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer sk_test_stub"},
    )
    # `payment_map or {...}` would silently ignore an explicitly-empty map,
    # which is exactly the case several tests need to exercise.
    gw._map = {"BK-47110": "pi_test_47110"} if payment_map is None else payment_map
    # Match the current on-disk mtime so the staleness check does not fire and
    # reload a real .stripe_payments.json over the fixture. Tests must not
    # depend on what happens to be lying in the working tree.
    from app.backend.payments import _map_mtime
    gw._map_mtime = _map_mtime()
    return gw


def refund_response(amount: int, **over):
    return {"id": "re_test_123", "object": "refund", "amount": amount,
            "currency": "gbp", "status": "succeeded", "livemode": False, **over}


# --- money -----------------------------------------------------------------

@pytest.mark.parametrize("gbp,pence", [
    (18.00, 1800), (22.50, 2250), (12.99, 1299), (0.01, 1), (1234.56, 123456),
    (18.30, 1830),   # int(18.30 * 100) == 1829 in float. The classic penny bug.
    (8.115, 812),    # half-up, not banker's rounding
])
def test_minor_units_are_exact(gbp, pence):
    assert to_minor_units(gbp) == pence


def test_a_cancellation_refunds_the_order_total_in_decimal():
    """22.50 must not take a detour through binary float before it is money."""
    assert refund_for_total(22.50) == Decimal("22.50")
    assert refund_for_total("22.50") == Decimal("22.50")
    assert to_minor_units(refund_for_total(22.50)) == 2250
    # The classic penny. Deriving from the total has to survive it too.
    assert refund_for_total(18.30) == Decimal("18.30")
    assert to_minor_units(refund_for_total(18.30)) == 1830


def test_zero_decimal_currencies_are_not_multiplied():
    assert to_minor_units(500, "jpy") == 500
    assert from_minor_units(500, "jpy") == 500.0


# --- the request Stripe actually receives ----------------------------------

def test_refund_sends_correct_amount_currency_and_metadata():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = dict(httpx.QueryParams(request.content.decode()))
        seen["idem"] = request.headers.get("idempotency-key")
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=refund_response(1800))

    out = stub_stripe(handler).refund(
        order_id="BK-47110", amount_gbp=18.00, rma="RMA-ABC123")

    assert seen["path"] == "/v1/refunds"
    assert seen["body"]["payment_intent"] == "pi_test_47110"
    assert seen["body"]["amount"] == "1800", "pence, not pounds"
    assert seen["body"]["metadata[rma]"] == "RMA-ABC123"
    assert seen["body"]["metadata[order_id]"] == "BK-47110"
    assert seen["auth"] == "Bearer sk_test_stub"
    assert out["refund_id"] == "re_test_123"
    assert out["amount_gbp"] == 18.00
    assert out["provider"] == "stripe"


def test_rma_is_the_idempotency_key():
    """The local ledger and Stripe are keyed on the same thing. That is the
    whole reason a double-click cannot double-refund."""
    seen = {}

    def handler(request):
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json=refund_response(1800))

    stub_stripe(handler).refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-XYZ")
    assert "RMA-XYZ" in seen["idem"]


def test_refund_uses_the_order_s_own_payment_intent_when_present():
    seen = {}

    def handler(request):
        seen["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(200, json=refund_response(1800))

    stub_stripe(handler, payment_map={}).refund(
        order_id="BK-47110", amount_gbp=18.0, rma="RMA-1",
        payment_intent_id="pi_from_order")
    assert seen["body"]["payment_intent"] == "pi_from_order"


# --- failure modes ---------------------------------------------------------

def test_unknown_order_raises_rather_than_guessing():
    gw = stub_stripe(lambda r: httpx.Response(200, json=refund_response(1)),
                     payment_map={})
    with pytest.raises(PaymentError, match="No Stripe payment on file"):
        gw.refund(order_id="BK-99999", amount_gbp=5.0, rma="RMA-1")


def test_stripe_error_becomes_payment_error():
    def handler(request):
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error",
            "message": "Charge ch_x has already been refunded."}})

    with pytest.raises(PaymentError, match="already been refunded"):
        stub_stripe(handler).refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")


def test_network_failure_becomes_payment_error():
    def handler(request):
        raise httpx.ConnectTimeout("timeout", request=request)

    with pytest.raises(PaymentError, match="unreachable"):
        stub_stripe(handler).refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")


def test_refund_post_is_never_auto_retried():
    """A POST that moves money must not be blindly repeated on a timeout —
    the first attempt may have succeeded."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectTimeout("timeout", request=request)

    with pytest.raises(PaymentError):
        stub_stripe(handler).refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")
    assert calls["n"] == 1, "exactly one attempt, unlike the read-only OMS path"


def test_failed_refund_status_raises():
    with pytest.raises(PaymentError, match="failed"):
        stub_stripe(lambda r: httpx.Response(
            200, json=refund_response(1800, status="failed"))
        ).refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")


def test_live_key_is_refused():
    with pytest.raises(RuntimeError, match="LIVE"):
        StripeGateway(api_key="sk_live_do_not_use")


# --- end to end through the agent ------------------------------------------

def test_agent_issues_a_real_stripe_refund():
    def handler(request):
        return httpx.Response(200, json=refund_response(1800, id="re_live_demo"))

    set_gateway(stub_stripe(handler))
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, it was damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    turn = agent.handle(s, "yes please")

    assert len(data.REFUNDS) == 1
    refund = next(iter(data.REFUNDS.values()))
    assert refund["refund_id"] == "re_live_demo"
    assert refund["provider"] == "stripe"
    assert "18.00" in turn.reply


def test_payment_failure_never_promises_money_to_the_customer():
    """The return is created, the refund fails. The customer must not be told
    an amount or a timeframe."""
    def handler(request):
        return httpx.Response(402, json={"error": {
            "type": "card_error", "message": "Insufficient funds for refund."}})

    set_gateway(stub_stripe(handler))
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, it was damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    turn = agent.handle(s, "yes please")

    assert len(data.REFUNDS) == 0, "no refund record for money that never moved"
    assert len(data.RETURNS) == 1, "the return itself did succeed"
    assert "£18.00" not in turn.reply, "must not promise an amount"
    assert "5 working days" not in turn.reply, "must not promise a timeframe"
    assert s.escalated, "a failed refund must reach a human"
    assert len(data.ESCALATIONS) == 1
    assert any(e["result"].startswith("failed:") for e in data.AUDIT_LOG
               if e["action"] == "issue_refund")


def test_double_confirmation_refunds_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=refund_response(1800))

    set_gateway(stub_stripe(handler))
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    agent.handle(s, "yes")
    agent.handle(s, "yes")

    assert len(data.REFUNDS) == 1
    assert calls["n"] == 1, "the local ledger short-circuits before the network"


def test_default_gateway_is_the_ledger(monkeypatch):
    monkeypatch.delenv("PAYMENTS_BACKEND", raising=False)
    set_gateway(None)
    from app.backend.payments import get_gateway
    assert isinstance(get_gateway(), LedgerGateway)


# --- the seed-after-start trap ---------------------------------------------

def test_gateway_rereads_the_payment_map_when_an_order_is_missing(tmp_path, monkeypatch):
    """Starting the app before seeding cached an empty map, and the failure only
    surfaced at the money step — after a return had already been created."""
    import app.backend.payments as P

    map_file = tmp_path / ".stripe_payments.json"
    monkeypatch.setattr(P, "PAYMENT_MAP_PATH", map_file)

    gw = stub_stripe(lambda r: httpx.Response(200, json=refund_response(1800)),
                     payment_map={})
    assert gw._map == {}, "started with nothing seeded"

    map_file.write_text(json.dumps({"BK-47110": "pi_seeded_after_startup"}))

    out = gw.refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")
    assert out["refund_id"] == "re_test_123"
    assert gw._map["BK-47110"] == "pi_seeded_after_startup"


def test_error_names_the_actual_cause_when_nothing_is_seeded(tmp_path, monkeypatch):
    import app.backend.payments as P
    monkeypatch.setattr(P, "PAYMENT_MAP_PATH", tmp_path / "absent.json")

    gw = stub_stripe(lambda r: httpx.Response(200, json=refund_response(1)),
                     payment_map={})
    with pytest.raises(PaymentError, match="payment map is empty"):
        gw.refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1")


def test_error_distinguishes_empty_map_from_a_missing_order(tmp_path, monkeypatch):
    import app.backend.payments as P
    map_file = tmp_path / ".stripe_payments.json"
    map_file.write_text(json.dumps({"BK-47110": "pi_x"}))
    monkeypatch.setattr(P, "PAYMENT_MAP_PATH", map_file)

    gw = stub_stripe(lambda r: httpx.Response(200, json=refund_response(1)),
                     payment_map={"BK-47110": "pi_x"})
    with pytest.raises(PaymentError, match="1 payment\\(s\\) are on file but none"):
        gw.refund(order_id="BK-99999", amount_gbp=5.0, rma="RMA-1")


# --- keeping the payment reference current ---------------------------------

def test_a_changed_map_is_picked_up_without_a_restart(tmp_path, monkeypatch):
    """Reload-on-miss was not enough: after a reset the map still HAS an entry
    for the order, just a spent one, so nothing missed and the process kept
    refunding against the PaymentIntent it emptied last run."""
    import time
    import app.backend.payments as P

    map_file = tmp_path / ".stripe_payments.json"
    map_file.write_text(json.dumps({"BK-47110": "pi_OLD"}))
    monkeypatch.setattr(P, "PAYMENT_MAP_PATH", map_file)

    gw = stub_stripe(lambda r: httpx.Response(200, json=refund_response(1800)),
                     payment_map={"BK-47110": "pi_OLD"})
    gw._map_mtime = P._map_mtime()
    assert gw._resolve_payment_intent("BK-47110", None) == "pi_OLD"

    time.sleep(0.01)
    map_file.write_text(json.dumps({"BK-47110": "pi_NEW"}))

    assert gw._resolve_payment_intent("BK-47110", None) == "pi_NEW", \
        "a reset must take effect without restarting the app"


def test_the_order_record_is_preferred_over_the_seeded_map():
    """The OMS is the system of record. The map is a demo convenience."""
    seen = {}

    def handler(request):
        seen["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(200, json=refund_response(1800))

    gw = stub_stripe(handler, payment_map={"BK-47110": "pi_from_map"})
    gw.refund(order_id="BK-47110", amount_gbp=18.0, rma="RMA-1",
              payment_intent_id="pi_from_the_order_record")
    assert seen["body"]["payment_intent"] == "pi_from_the_order_record"


def test_an_oms_carried_payment_flows_all_the_way_through(monkeypatch):
    """End to end: a payment_intent_id on the order reaches Stripe. This is the
    path that needs no cache invalidation at all — the order is re-fetched on
    every refund, so a reset takes effect immediately."""
    from app.backend import data as d
    seen = {}

    def handler(request):
        seen["pi"] = dict(httpx.QueryParams(request.content.decode()))["payment_intent"]
        return httpx.Response(200, json=refund_response(1800))

    set_gateway(stub_stripe(handler, payment_map={}))
    monkeypatch.setitem(d.ORDERS["BK-47110"], "payment_intent_id", "pi_on_the_order")

    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    agent.handle(s, "yes")

    assert seen["pi"] == "pi_on_the_order"
    assert len(data.REFUNDS) == 1
