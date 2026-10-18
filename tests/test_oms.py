"""Tests for the OMS seam.

The point of these is that they run offline. A test that needs mockapi.io to be
up is not testing your code, it is testing someone else's uptime - so the HTTP
adapter is exercised against a stubbed transport instead, and what gets asserted
is the mapping and the failure behaviour.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.backend import data
from app.backend.oms import BooklyHttpOMS, InMemoryOMS, OmsUnavailable, set_oms
from app.memory import Session
from app.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def restore_oms():
    yield
    set_oms(None)
    data.RETURNS.clear(); data.REFUNDS.clear()
    data.ESCALATIONS.clear(); data.AUDIT_LOG.clear()
    data.CANCELLATIONS.clear()


def stub_oms(handler) -> BooklyHttpOMS:
    """A real BooklyHttpOMS with its transport swapped for a stub."""
    oms = BooklyHttpOMS.__new__(BooklyHttpOMS)
    oms.base = "https://stub.mockapi.io/api/v1"
    oms._client = httpx.Client(
        base_url=oms.base, transport=httpx.MockTransport(handler)
    )
    return oms


# --- mapping ---------------------------------------------------------------

def test_http_adapter_maps_mockapi_records_to_canonical_shape():
    """mockapi adds its own `id` and can return numbers as strings."""
    raw = {
        "id": "7",                                   # mockapi surrogate key
        "order_id": "BK-99001",
        "customer_id": "cus_1001",
        "placed_at": "2026-09-01T10:00:00+00:00",
        "delivered_at": "2026-09-03T10:00:00+00:00",
        "status": "delivered",
        "total_gbp": "24.50",                        # string, not float
        "tracking_id": "",                           # empty, not null
        "items": [{"sku": 9780099540946, "title": "Wolf Hall",
                   "qty": "1", "price_gbp": "24.50"}],
    }

    def handler(request):
        assert request.url.params["order_id"] == "BK-99001"
        return httpx.Response(200, json=[raw])

    order = stub_oms(handler).get_order("bk-99001")   # lowercase input

    assert order["order_id"] == "BK-99001"
    assert order["total_gbp"] == 24.50 and isinstance(order["total_gbp"], float)
    assert order["tracking_id"] is None
    assert order["items"][0]["sku"] == "9780099540946"
    assert order["items"][0]["qty"] == 1
    assert "id" not in order, "mockapi's surrogate id must not leak inwards"


def test_http_adapter_looks_up_by_business_key_not_surrogate_id():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    stub_oms(handler).get_order("BK-48291")
    assert seen["path"].endswith("/orders")
    assert seen["params"] == {"order_id": "BK-48291"}


def test_missing_record_is_none_not_an_error():
    assert stub_oms(lambda r: httpx.Response(200, json=[])).get_order("BK-00000") is None
    assert stub_oms(lambda r: httpx.Response(404)).get_order("BK-00000") is None


def test_orders_come_back_newest_first():
    rows = [
        {"order_id": "BK-1", "customer_id": "c", "placed_at": "2026-01-01T00:00:00+00:00",
         "status": "delivered", "items": []},
        {"order_id": "BK-2", "customer_id": "c", "placed_at": "2026-06-01T00:00:00+00:00",
         "status": "delivered", "items": []},
    ]
    oms = stub_oms(lambda r: httpx.Response(200, json=rows))
    assert [o["order_id"] for o in oms.list_orders("c")] == ["BK-2", "BK-1"]


# --- failure behaviour -----------------------------------------------------

def test_transport_failure_retries_once_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(OmsUnavailable):
        stub_oms(handler).get_order("BK-48291")
    assert calls["n"] == 2, "one retry, not zero and not a storm"


def test_transient_failure_recovers_on_the_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("cold start", request=request)
        return httpx.Response(200, json=[])
    assert stub_oms(handler).get_order("BK-48291") is None


def test_server_error_becomes_oms_unavailable():
    with pytest.raises(OmsUnavailable):
        stub_oms(lambda r: httpx.Response(500, text="nope")).get_order("BK-48291")


def test_oms_outage_degrades_to_a_human_handoff():
    """The whole point of raising rather than returning None: an OMS outage
    must reach the customer as an honest escalation, not a wrong answer."""
    def dead(request):
        raise httpx.ConnectTimeout("timeout", request=request)

    set_oms(stub_oms(dead))
    agent = Orchestrator("mock")
    s = Session()
    turn = agent.handle(s, "Where's my order?")
    turn = agent.handle(s, "rich@example.com")

    assert any(st["status"] == "error" for st in turn.trace)
    assert isinstance(turn.reply, str) and turn.reply
    assert len(data.REFUNDS) == 0


# --- the two backends agree ------------------------------------------------

def test_both_backends_return_the_same_canonical_shape():
    """Contract test: whatever the source, the agent sees one shape."""
    mem = InMemoryOMS()
    order = mem.get_order("BK-47110")

    def handler(request):
        return httpx.Response(200, json=[{"id": "1", **order}])

    assert json.dumps(stub_oms(handler).get_order("BK-47110"), sort_keys=True) == \
           json.dumps(order, sort_keys=True)


def test_http_cancel_puts_once_and_does_not_retry():
    """The status write is one PUT. The refund POST is a different call and
    is not made here; this test pins the OMS half of that promise."""
    puts = {"n": 0}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=[{
                "id": "9",
                "order_id": "BK-48307",
                "customer_id": "cus_1001",
                "placed_at": "2026-09-01T10:00:00+00:00",
                "status": "processing",
                "total_gbp": "22.50",
                "items": [],
            }])
        if request.method == "PUT":
            puts["n"] += 1
            assert request.url.path.endswith("/orders/9")
            return httpx.Response(200, json={
                "id": "9",
                "order_id": "BK-48307",
                "customer_id": "cus_1001",
                "placed_at": "2026-09-01T10:00:00+00:00",
                "status": "cancelled",
                "total_gbp": "22.50",
                "items": [],
            })
        raise AssertionError(request.method)

    order = stub_oms(handler).cancel_order("bk-48307")
    assert puts["n"] == 1
    assert order["status"] == "cancelled"
    assert order["order_id"] == "BK-48307"


def test_default_backend_is_in_memory(monkeypatch):
    monkeypatch.delenv("OMS_BACKEND", raising=False)
    set_oms(None)
    from app.backend.oms import get_oms
    assert get_oms().name == "memory"
