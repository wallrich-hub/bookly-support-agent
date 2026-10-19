"""Order Management System: one interface, two implementations.

Why an interface rather than just swapping the calls out:

  * the demo and the test suite stay offline, deterministic and fast, which is
    what lets the scenario tests assert architectural properties in 0.06s
  * a live third-party dependency cannot take the demo down mid-review
  * the seam is exactly where a real customer's OMS would be plugged in, which
    is the honest answer to "how would this work against our systems?"

`BooklyHttpOMS` talks to mockapi.io over real HTTP - real latency, real
timeouts, real failure modes - and maps its records into the canonical shape
the rest of the agent expects. That mapping is deliberate: nothing outside this
module knows what the upstream payload looks like, so replacing mockapi.io with
a customer's Commerce Cloud or a homegrown OMS touches one file.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Protocol

from . import data, wire


def _summarise(status: int, response) -> str:
    """One readable line about what came back, without dumping the payload."""
    if status >= 400:
        return f"HTTP {status}"
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {status}, non-JSON"
    if isinstance(body, list):
        return f"{len(body)} record{'s' if len(body) != 1 else ''}"
    if isinstance(body, dict):
        return ", ".join(f"{k}={body[k]}" for k in list(body)[:3])
    return str(body)[:80]


class OmsUnavailable(RuntimeError):
    """Upstream is unreachable or misbehaving.

    Raised, never swallowed. The tool registry turns it into a tool error, and
    the orchestrator turns that into a human handoff - so an OMS outage
    degrades into an honest escalation instead of a wrong answer.
    """


class OrderSource(Protocol):
    name: str

    def get_customer_by_email(self, email: str) -> dict[str, Any] | None: ...
    def list_orders(self, customer_id: str) -> list[dict[str, Any]]: ...
    def get_order(self, order_id: str) -> dict[str, Any] | None: ...
    def get_shipment(self, tracking_id: str) -> dict[str, Any] | None: ...
    def cancel_order(self, order_id: str) -> dict[str, Any] | None: ...


# =====================================================================
# In-memory (default)
# =====================================================================
class InMemoryOMS:
    name = "memory"

    def get_customer_by_email(self, email):
        with wire.local("oms:memory", f"get_customer_by_email({email})") as call:
            email = (email or "").strip().lower()
            found = next(
                (c for c in data.CUSTOMERS.values() if c["email"].lower() == email), None)
            call.done(summary=f"{found['customer_id']}" if found else "no match")
            return found

    def list_orders(self, customer_id):
        with wire.local("oms:memory", f"list_orders({customer_id})") as call:
            rows = [self._view(o) for o in data.ORDERS.values()
                    if o["customer_id"] == customer_id]
            rows.sort(key=lambda o: o["placed_at"], reverse=True)
            call.done(summary=f"{len(rows)} orders")
            return rows

    def get_order(self, order_id):
        with wire.local("oms:memory", f"get_order({order_id})") as call:
            raw = data.ORDERS.get((order_id or "").strip().upper())
            order = self._view(raw)
            call.done(summary=order["status"] if order else "not found")
            return order

    def cancel_order(self, order_id):
        """Mark an order cancelled. Idempotent: a second call returns the same row.

        The seed catalogue is not rewritten. The cancellation lives in
        CANCELLATIONS and `_view` applies it, so a reset drops the overlay
        and the order is processing again.
        """
        key = (order_id or "").strip().upper()
        with wire.local("oms:memory", f"cancel_order({key})") as call:
            if key not in data.ORDERS:
                call.done(summary="not found")
                return None
            if key not in data.CANCELLATIONS:
                data.CANCELLATIONS[key] = {
                    "order_id": key,
                    "status": "cancelled",
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                }
            call.done(summary="cancelled")
            return self._view(data.ORDERS[key])

    def get_shipment(self, tracking_id):
        with wire.local("oms:memory", f"get_shipment({tracking_id})") as call:
            ship = data.SHIPMENTS.get(tracking_id)
            call.done(summary=ship["status"] if ship else "not found")
            return ship

    @staticmethod
    def _view(order: dict | None) -> dict | None:
        if order is None:
            return None
        viewed = {**order, "payment_intent_id": order.get("payment_intent_id")}
        cancel = data.CANCELLATIONS.get(viewed["order_id"])
        if cancel:
            viewed["status"] = "cancelled"
            viewed["cancelled_at"] = cancel["cancelled_at"]
        return viewed


# =====================================================================
# mockapi.io over HTTP
# =====================================================================
class BooklyHttpOMS:
    """REST client for a mockapi.io project.

    mockapi.io conventions used here:
      GET {base}/{resource}                 -> list
      GET {base}/{resource}?field=value     -> filter by exact field match
    Note that mockapi assigns its own surrogate `id` to every record, which is
    NOT our business key. We always look orders up by `order_id`, never by `id`.
    """

    name = "mockapi"

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        import httpx  # imported lazily so the memory backend needs no dependency

        base = (base_url or os.environ.get("OMS_BASE_URL") or "").rstrip("/")
        if not base:
            raise RuntimeError(
                "OMS_BACKEND=http requires OMS_BASE_URL, e.g. "
                "https://<token>.mockapi.io/api/v1"
            )
        self.base = base
        self._client = httpx.Client(
            base_url=base,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    # --- transport -----------------------------------------------------
    def _get(self, path: str, **params) -> Any:
        """One bounded retry, then give up loudly.

        Retrying once absorbs a cold start or a dropped connection. Retrying
        forever just moves the customer's wait from the OMS to us, so the
        second failure becomes an escalation instead.
        """
        import httpx

        last: Exception | None = None
        for attempt in range(2):
            label = path if attempt == 0 else f"{path}  (retry)"
            with wire.span("oms:mockapi", "GET", label, params=params) as call:
                try:
                    r = self._client.get(path, params=params or None)
                    call.done(status=r.status_code,
                              summary=_summarise(r.status_code, r))
                    if r.status_code == 404:
                        return None
                    r.raise_for_status()
                    return r.json()
                except (httpx.HTTPError, ValueError) as exc:
                    last = exc
                    call.note(outcome="error", status=getattr(
                        getattr(exc, "response", None), "status_code", None))
                    call.done(summary=f"{type(exc).__name__}: {exc}")
        raise OmsUnavailable(f"GET {path} failed: {type(last).__name__}: {last}")

    @staticmethod
    def _first(rows: Any) -> dict[str, Any] | None:
        if isinstance(rows, list):
            return rows[0] if rows else None
        return rows if isinstance(rows, dict) else None

    # --- anti-corruption layer -----------------------------------------
    # Upstream shapes change; the agent's shape does not. Everything crossing
    # this boundary is coerced, defaulted and renamed here.
    @staticmethod
    def _customer(r: dict | None) -> dict[str, Any] | None:
        if not r:
            return None
        return {
            "customer_id": r["customer_id"],
            "email": r["email"],
            "name": r.get("name", ""),
            "tier": r.get("tier", "standard"),
            "postcode": r.get("postcode", ""),
            "lifetime_orders": int(r.get("lifetime_orders", 0)),
            "refund_count_90d": int(r.get("refund_count_90d", 0)),
        }

    @staticmethod
    def _order(r: dict | None) -> dict[str, Any] | None:
        if not r:
            return None
        items = r.get("items") or []
        if isinstance(items, str):          # some seeders flatten nested arrays
            import json
            try:
                items = json.loads(items)
            except ValueError:
                items = []
        order = {
            "order_id": r["order_id"],
            "customer_id": r["customer_id"],
            "placed_at": r["placed_at"],
            "status": r.get("status", "processing"),
            "total_gbp": float(r.get("total_gbp", 0) or 0),
            "tracking_id": r.get("tracking_id") or None,
            "payment_intent_id": r.get("payment_intent_id") or None,
            "items": [
                {
                    "sku": str(i["sku"]),
                    "title": i.get("title", ""),
                    "qty": int(i.get("qty", 1)),
                    "price_gbp": float(i.get("price_gbp", 0) or 0),
                }
                for i in items
            ],
        }
        if r.get("delivered_at"):
            order["delivered_at"] = r["delivered_at"]
        return order

    @staticmethod
    def _shipment(r: dict | None) -> dict[str, Any] | None:
        if not r:
            return None
        return {
            "tracking_id": r["tracking_id"],
            "carrier": r.get("carrier", "Carrier"),
            "status": r.get("status", "in_transit"),
            "last_scan": r.get("last_scan", ""),
            "last_scan_at": r.get("last_scan_at", ""),
            "expected_delivery": r.get("expected_delivery", ""),
        }

    # --- OrderSource ----------------------------------------------------
    def get_customer_by_email(self, email):
        rows = self._get("/customers", email=(email or "").strip().lower())
        return self._customer(self._first(rows))

    def list_orders(self, customer_id):
        rows = self._get("/orders", customer_id=customer_id) or []
        orders = [o for o in (self._order(r) for r in rows) if o]
        return sorted(orders, key=lambda o: o["placed_at"], reverse=True)

    def get_order(self, order_id):
        rows = self._get("/orders", order_id=(order_id or "").strip().upper())
        return self._order(self._first(rows))

    def get_shipment(self, tracking_id):
        rows = self._get("/shipments", tracking_id=tracking_id)
        return self._shipment(self._first(rows))

    def cancel_order(self, order_id):
        """PUT the order to cancelled. One attempt, no retry.

        A status write is not money, but retrying it blindly is still the
        wrong habit next to the refund this cancellation is about to make.
        The read above may retry once; this PUT does not.
        """
        import httpx

        key = (order_id or "").strip().upper()
        raw = self._first(self._get("/orders", order_id=key))
        if not raw or not raw.get("id"):
            return None
        path = f"/orders/{raw['id']}"
        with wire.span("oms:mockapi", "PUT", path, params={"status": "cancelled"}) as call:
            try:
                response = self._client.put(path, json={"status": "cancelled"})
                call.done(status=response.status_code,
                          summary=_summarise(response.status_code, response))
                if response.status_code >= 400:
                    return None
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                call.note(outcome="error")
                call.done(summary=f"{type(exc).__name__}: {exc}")
                return None
        if not isinstance(body, dict):
            body = {**raw, "status": "cancelled"}
        return self._order(body)


# =====================================================================
_OMS: OrderSource | None = None


def get_oms() -> OrderSource:
    global _OMS
    if _OMS is None:
        backend = os.environ.get("OMS_BACKEND", "memory").lower()
        _OMS = BooklyHttpOMS() if backend in ("http", "mockapi") else InMemoryOMS()
    return _OMS


def set_oms(source: OrderSource | None) -> None:
    """Test seam."""
    global _OMS
    _OMS = source
