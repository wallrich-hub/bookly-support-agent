"""Seed data for the mocked Bookly systems of record.

In a real deployment these rows live in OMS / WMS / Stripe / Zendesk. They are
in-memory here so the prototype runs with zero infrastructure.
"""

from __future__ import annotations

from datetime import timedelta

from . import clock

# Anchored to real time at import, so the seeded return windows stay correct
# however long after this was written the demo is run.
NOW = clock.now().replace(microsecond=0)   # whole seconds: tidier seeded data


def days_ago(n: int) -> str:
    return (NOW - timedelta(days=n)).isoformat()


# --- Customers (CRM) -------------------------------------------------------
CUSTOMERS = {
    "cus_1001": {
        "customer_id": "cus_1001",
        "email": "rich@example.com",
        "name": "Rich Wall",
        "tier": "plus",  # plus members get a 60-day return window
        "postcode": "CM1 2AB",
        "lifetime_orders": 14,
        "refund_count_90d": 0,
    },
    "cus_1002": {
        "customer_id": "cus_1002",
        "email": "sam@example.com",
        "name": "Sam Okafor",
        "tier": "standard",
        "postcode": "SS9 1AA",
        "lifetime_orders": 2,
        "refund_count_90d": 3,  # triggers the abuse guardrail
    },
}

# --- Orders (OMS) ----------------------------------------------------------
# cus_1001 deliberately has TWO open orders so "where is my order?" is
# genuinely ambiguous and the agent is forced to ask a clarifying question.
ORDERS = {
    "BK-48291": {
        "order_id": "BK-48291",
        "customer_id": "cus_1001",
        "placed_at": days_ago(4),
        "status": "shipped",
        "total_gbp": 34.98,
        "tracking_id": "TRK-77120",
        "items": [
            {"sku": "9780571366897", "title": "Piranesi", "qty": 1, "price_gbp": 16.99},
            {"sku": "9780241988268", "title": "Klara and the Sun", "qty": 1, "price_gbp": 17.99},
        ],
    },
    "BK-48307": {
        "order_id": "BK-48307",
        "customer_id": "cus_1001",
        "placed_at": days_ago(2),
        "status": "processing",
        "total_gbp": 22.50,
        "tracking_id": None,
        "items": [
            {"sku": "9781784742324", "title": "The Overstory", "qty": 1, "price_gbp": 22.50},
        ],
    },
    "BK-47110": {
        "order_id": "BK-47110",
        "customer_id": "cus_1001",
        "placed_at": days_ago(45),
        "status": "delivered",
        "delivered_at": days_ago(40),
        "total_gbp": 18.00,
        "tracking_id": "TRK-71004",
        "items": [
            {"sku": "9780099540946", "title": "Wolf Hall", "qty": 1, "price_gbp": 18.00},
        ],
    },
    "BK-46002": {
        "order_id": "BK-46002",
        "customer_id": "cus_1001",
        "placed_at": days_ago(150),
        "status": "delivered",
        "delivered_at": days_ago(145),
        "total_gbp": 12.99,
        "tracking_id": "TRK-66001",
        "items": [
            {"sku": "9780140449136", "title": "Meditations", "qty": 1, "price_gbp": 12.99},
        ],
    },
    "BK-48800": {
        "order_id": "BK-48800",
        "customer_id": "cus_1002",
        "placed_at": days_ago(10),
        "status": "delivered",
        "delivered_at": days_ago(6),
        "total_gbp": 29.99,
        "tracking_id": "TRK-78222",
        "items": [
            {"sku": "9781787330672", "title": "Circe", "qty": 1, "price_gbp": 29.99},
        ],
    },
}

# --- Shipments (carrier) ---------------------------------------------------
SHIPMENTS = {
    "TRK-77120": {
        "tracking_id": "TRK-77120",
        "carrier": "Royal Mail",
        "status": "in_transit",
        "last_scan": "Departed Chelmsford Mail Centre",
        "last_scan_at": days_ago(1),
        "expected_delivery": (NOW + timedelta(days=1)).date().isoformat(),
    },
    "TRK-71004": {
        "tracking_id": "TRK-71004",
        "carrier": "Evri",
        "status": "delivered",
        "last_scan": "Delivered - left in porch",
        "last_scan_at": days_ago(40),
        "expected_delivery": (NOW - timedelta(days=40)).date().isoformat(),
    },
    "TRK-66001": {
        "tracking_id": "TRK-66001",
        "carrier": "Evri",
        "status": "delivered",
        "last_scan": "Delivered - signed for",
        "last_scan_at": days_ago(145),
        "expected_delivery": (NOW - timedelta(days=145)).date().isoformat(),
    },
    "TRK-78222": {
        "tracking_id": "TRK-78222",
        "carrier": "Royal Mail",
        "status": "delivered",
        "last_scan": "Delivered - handed to resident",
        "last_scan_at": days_ago(6),
        "expected_delivery": (NOW - timedelta(days=6)).date().isoformat(),
    },
}

# Side-effect ledgers. Every write the agent makes lands here so the demo can
# show that actions are real, idempotent and auditable.
# REFUNDS is written only by bookly_api.issue_refund, after the gateway
# succeeds. A sentence in the chat does not create a row.
# ESCALATIONS keep the transcript passed into create_escalation, for a person
# to read. That text is untrusted and is not a source of identity, amounts,
# or permission.
RETURNS: dict[str, dict] = {}
REFUNDS: dict[str, dict] = {}
ESCALATIONS: dict[str, dict] = {}
PASSWORD_RESETS: list[dict] = []
AUDIT_LOG: list[dict] = []
# order_id -> cancellation record. Kept off the seed rows so a cancel in one
# test cannot change BK-48307 for the next, and so the reset endpoint can
# put the demo back without rewriting the catalogue.
CANCELLATIONS: dict[str, dict] = {}
