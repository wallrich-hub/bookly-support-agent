"""Egress log tests.

The trace is shown on screen and kept as an audit record, so the two things
that matter are that it captures every outbound call, and that it never
captures a credential.
"""

from __future__ import annotations

import httpx
import pytest

from app.backend import data, wire
from app.backend.payments import set_gateway
from app.memory import Session
from app.orchestrator import Orchestrator
from tests.test_payments import refund_response, stub_stripe


@pytest.fixture(autouse=True)
def clean():
    yield
    set_gateway(None)
    data.RETURNS.clear(); data.REFUNDS.clear()
    data.ESCALATIONS.clear(); data.AUDIT_LOG.clear()


# --- redaction -------------------------------------------------------------

def test_credentials_are_never_recorded():
    wire.begin()
    with wire.span("payments:stripe", "POST", "/refunds", params={
            "amount": 1800,
            "Authorization": "Bearer sk_test_supersecret",
            "api_key": "sk_test_supersecret",
            "card[number]": "4242424242424242",
    }) as call:
        call.done(status=200, summary="re_1 succeeded")

    entry = wire.drain()[0]
    blob = repr(entry)
    assert "supersecret" not in blob
    assert "4242424242424242" not in blob
    assert entry["params"]["amount"] == 1800, "non-sensitive fields survive"


def test_long_payloads_are_truncated():
    wire.begin()
    with wire.span("oms:mockapi", "GET", "/orders") as call:
        call.done(status=200, summary="x" * 5000)
    assert len(wire.drain()[0]["summary"]) <= wire.MAX_PREVIEW + 1


# --- capture ---------------------------------------------------------------

def test_span_records_timing_status_and_outcome():
    wire.begin()
    with wire.span("oms:mockapi", "GET", "/orders", params={"order_id": "BK-1"}) as call:
        call.done(status=200, summary="1 record")
    entry = wire.drain()[0]
    assert entry["outcome"] == "ok" and entry["status"] == 200
    assert entry["method"] == "GET" and entry["path"] == "/orders"
    assert entry["ms"] >= 0


def test_an_exception_marks_the_call_failed_and_still_records_it():
    wire.begin()
    with pytest.raises(ValueError):
        with wire.span("oms:mockapi", "GET", "/orders"):
            raise ValueError("connection reset")
    entry = wire.drain()[0]
    assert entry["outcome"] == "error"
    assert "connection reset" in entry["summary"]


def test_http_status_400_is_an_error_outcome():
    wire.begin()
    with wire.span("payments:stripe", "POST", "/refunds") as call:
        call.done(status=402, summary="card_error")
    assert wire.drain()[0]["outcome"] == "error"


# --- through the agent -----------------------------------------------------

def test_every_trace_step_carries_the_calls_it_made():
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "Where's my order?")
    turn = agent.handle(s, "rich@example.com")

    steps = {st["tool"]: st for st in turn.trace}
    assert steps["identify_customer"]["calls"], "a tool with no calls tells you nothing"
    assert steps["identify_customer"]["calls"][0]["system"] == "oms:memory"
    assert steps["list_my_orders"]["calls"][0]["summary"] == "4 orders"


def test_calls_are_scoped_to_the_tool_that_made_them():
    """A tool's calls must not leak into the next tool's trace entry."""
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, damaged")
    agent.handle(s, "rich@example.com")
    turn = agent.handle(s, "BK-47110")

    by_tool = {st["tool"]: st["calls"] for st in turn.trace}
    assert all(c["path"].startswith("get_order") or c["path"].startswith("get_shipment")
               for c in by_tool["get_order_status"])
    assert len(by_tool["check_return_eligibility"]) == 1


def test_stripe_call_appears_in_the_trace_with_amount_and_idempotency_key():
    """The two fields that make the integration defensible must be visible."""
    set_gateway(stub_stripe(lambda r: httpx.Response(200, json=refund_response(1800))))
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    turn = agent.handle(s, "yes")

    stripe_calls = [c for st in turn.trace for c in st["calls"]
                    if c["system"] == "payments:stripe"]
    assert len(stripe_calls) == 1
    call = stripe_calls[0]
    assert call["method"] == "POST" and call["path"] == "/refunds"
    assert call["params"]["amount"] == 1800, "minor units, visible on screen"
    assert "RMA-" in call["params"]["Idempotency-Key"]
    assert "succeeded" in call["summary"]
    assert "Bearer" not in repr(call)


def test_a_failed_call_is_still_traced():
    set_gateway(stub_stripe(lambda r: httpx.Response(402, json={"error": {
        "type": "card_error", "message": "Insufficient funds."}})))
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "I want to return a book, damaged")
    agent.handle(s, "rich@example.com")
    agent.handle(s, "BK-47110")
    turn = agent.handle(s, "yes")

    failed = [c for st in turn.trace for c in st["calls"] if c["outcome"] == "error"]
    assert failed, "the failure must be visible in the trace, not just in the reply"
    assert failed[0]["status"] == 402


# --- the trace is for humans, not for the model ---------------------------

def test_wire_log_is_not_sent_to_the_model():
    """Instrumentation must not reach the model's context: it costs tokens and
    invites reasoning about HTTP status codes instead of the customer."""
    agent = Orchestrator("mock")
    s = Session()
    agent.handle(s, "Where's my order?")
    turn = agent.handle(s, "rich@example.com")

    assert turn.trace[0]["calls"], "the trace still has them"
    tool_msgs = [m for m in s.working if m["role"] == "tool"]
    assert tool_msgs
    for m in tool_msgs:
        assert "calls" not in m["content"], "wire log leaked into the LLM context"
