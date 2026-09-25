#!/usr/bin/env python3
"""Seed a mockapi.io project with the Bookly demo data.

    python scripts/seed_mockapi.py --base https://<token>.mockapi.io/api/v1 --wipe
    python scripts/seed_mockapi.py --base https://<token>.mockapi.io/api/v1 --check

Create three resources in your mockapi.io project first, named exactly:
    customers   orders   shipments

Dates are written relative to the moment you run this, so the return-window
demo is correct on the day you seed it. Re-run before recording - a 40-day-old
delivery seeded six weeks ago is outside the window and your "eligible" case
will have silently become an "escalate" case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load as _load_env  # noqa: E402

_load_env()   # .env, so a new terminal tab is not a fresh start

import httpx  # noqa: E402

from app.backend import data  # noqa: E402

RESOURCES = ("customers", "orders", "shipments")


def rows_for(resource: str) -> list[dict]:
    if resource == "customers":
        return list(data.CUSTOMERS.values())
    if resource == "orders":
        return list(data.ORDERS.values())
    return list(data.SHIPMENTS.values())


def wipe(client: httpx.Client, resource: str) -> int:
    """mockapi has no bulk delete, so enumerate and remove by its surrogate id."""
    existing = client.get(f"/{resource}")
    if existing.status_code == 404:
        return 0
    existing.raise_for_status()
    rows = existing.json() or []
    for row in rows:
        client.delete(f"/{resource}/{row['id']}").raise_for_status()
    return len(rows)


def seed(base: str, do_wipe: bool) -> None:
    with httpx.Client(base_url=base.rstrip("/"), timeout=20.0) as client:
        for resource in RESOURCES:
            if do_wipe:
                removed = wipe(client, resource)
                print(f"  {resource}: removed {removed}")
            for row in rows_for(resource):
                r = client.post(f"/{resource}", json=row)
                if r.status_code >= 400:
                    print(f"  ! {resource} {row}: {r.status_code} {r.text[:160]}")
                    r.raise_for_status()
            print(f"  {resource}: seeded {len(rows_for(resource))}")


def _comparable(order: dict) -> dict:
    """Strip what is legitimately allowed to differ between two runs.

    Timestamps are anchored to import time, so the seeding process and this one
    stamp different instants - and if you seed today and check next week they
    differ by days. Comparing raw ISO strings would fail forever for no reason.
    What must match is the shape, the business fields, and the DAY.
    """
    out = dict(order)
    for field in ("placed_at", "delivered_at"):
        if out.get(field):
            out[field] = out[field][:10]        # date only
    return out


def check(base: str) -> int:
    """Round-trip the data through the real adapter and verify three things.

    Not "did the POST return 201" - that proves nothing. What matters is
    whether the agent's OMS interface returns the same canonical shape, and
    whether the policy engine reaches the same DECISION against it. The second
    is the one that predicts whether the demo behaves identically.
    """
    import os

    os.environ["OMS_BACKEND"] = "http"
    os.environ["OMS_BASE_URL"] = base
    from app.backend.oms import BooklyHttpOMS, InMemoryOMS
    from app.policy import engine

    http, mem = BooklyHttpOMS(base), InMemoryOMS()
    failures = 0

    for order_id in data.ORDERS:
        a, b = mem.get_order(order_id), http.get_order(order_id)
        if b is None:
            print(f"  ✗ {order_id}: not found upstream")
            failures += 1
            continue

        if set(a) != set(b):
            missing, extra = set(a) - set(b), set(b) - set(a)
            print(f"  ✗ {order_id}: fields differ (missing {missing or '-'}, "
                  f"unexpected {extra or '-'})")
            failures += 1
            continue

        if json.dumps(_comparable(a), sort_keys=True) != \
           json.dumps(_comparable(b), sort_keys=True):
            print(f"  ✗ {order_id}: values differ")
            print(f"      memory: {json.dumps(_comparable(a), sort_keys=True)[:220]}")
            print(f"      http:   {json.dumps(_comparable(b), sort_keys=True)[:220]}")
            failures += 1
            continue

        # The check that actually predicts demo behaviour: same data in, same
        # policy decision out.
        customer = mem.get_customer_by_email(
            next(c["email"] for c in data.CUSTOMERS.values()
                 if c["customer_id"] == a["customer_id"]))
        sku = a["items"][0]["sku"] if a["items"] else None
        if sku:
            da = engine.evaluate_return(a, customer, sku)
            db = engine.evaluate_return(b, customer, sku)
            if (da.code, da.facts.get("refund_amount_gbp")) != \
               (db.code, db.facts.get("refund_amount_gbp")):
                print(f"  ✗ {order_id}: policy decision differs "
                      f"({da.code} vs {db.code})")
                failures += 1
                continue
            print(f"  ✓ {order_id}  → policy: {db.code}")
        else:
            print(f"  ✓ {order_id}")

    for email in (c["email"] for c in data.CUSTOMERS.values()):
        if http.get_customer_by_email(email) is None:
            print(f"  ✗ customer {email}: not found upstream")
            failures += 1
        else:
            print(f"  ✓ customer {email}")

    cus = next(iter(data.CUSTOMERS))
    n = len(http.list_orders(cus))
    expected = len(mem.list_orders(cus))
    print(f"  {'✓' if n == expected else '✗'} list_orders({cus}) -> {n} (expected {expected})")
    failures += 0 if n == expected else 1

    return failures


def probe(base: str) -> int:
    """Write one throwaway order, read it back, report what survived, delete it.

    The open question with any hosted mock is whether it preserves arbitrary
    fields and nested arrays, or flattens them to match a UI-defined schema.
    Rather than assume, find out against the actual project in 30 seconds -
    before seeding eleven records on top of a wrong assumption.
    """
    canary = {
        "order_id": "BK-00000",
        "customer_id": "probe",
        "placed_at": "2026-01-01T00:00:00+00:00",
        "status": "delivered",
        "total_gbp": 1.23,
        "tracking_id": "TRK-PROBE",
        "items": [{"sku": "1234567890123", "title": "Probe",
                   "qty": 1, "price_gbp": 1.23}],
    }

    with httpx.Client(base_url=base.rstrip("/"), timeout=20.0) as client:
        r = client.post("/orders", json=canary)
        if r.status_code >= 400:
            print(f"  ✗ POST /orders -> {r.status_code} {r.text[:200]}")
            print("\n    Does a resource named exactly 'orders' exist in the project?")
            return 1
        created = r.json()
        rid = created.get("id")
        print(f"  ✓ POST accepted, mockapi id={rid}")

        back = client.get("/orders", params={"order_id": "BK-00000"})
        back.raise_for_status()
        rows = back.json() or []
        if not rows:
            print("  ✗ filtering by ?order_id= returned nothing")
            print("\n    The adapter looks orders up by business key. If filtering")
            print("    is unsupported, that is the thing to fix before seeding.")
            if rid:
                client.delete(f"/orders/{rid}")
            return 1
        got = rows[0]
        print("  ✓ ?order_id= filtering works")

        problems = 0
        for field, expected in canary.items():
            if field not in got:
                print(f"  ✗ field '{field}' was dropped")
                problems += 1
            elif field == "items":
                items = got["items"]
                if isinstance(items, str):
                    print("  ! 'items' came back as a JSON *string*, not an array")
                    print("      -> harmless: the adapter already parses that case")
                elif not isinstance(items, list) or not items:
                    print(f"  ✗ 'items' came back as {type(items).__name__}: {items!r}")
                    problems += 1
                elif not isinstance(items[0], dict) or "sku" not in items[0]:
                    print(f"  ✗ nested item objects were mangled: {items[0]!r}")
                    problems += 1
                else:
                    print("  ✓ nested 'items' array survived intact")
            elif str(got[field]) != str(expected):
                print(f"  ! '{field}' changed type: sent {expected!r}, got {got[field]!r}")
                print("      -> harmless: the adapter coerces this")

        if rid:
            client.delete(f"/orders/{rid}")
            print(f"  ✓ cleaned up probe record {rid}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="e.g. https://<token>.mockapi.io/api/v1")
    ap.add_argument("--probe", action="store_true",
                    help="write/read/delete one throwaway record to find out what "
                         "this project actually preserves. Run this first.")
    ap.add_argument("--wipe", action="store_true",
                    help="delete existing records first (recommended - reseeding "
                         "without this duplicates every row)")
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not write")
    args = ap.parse_args()

    if args.probe:
        print(f"Probing {args.base}\n")
        problems = probe(args.base)
        print("\nLooks good — safe to seed." if not problems
              else f"\n{problems} problem(s). Fix before seeding.")
        return 1 if problems else 0

    if args.check:
        print(f"Checking {args.base}")
        failures = check(args.base)
        print("\nAll good." if not failures else f"\n{failures} problem(s).")
        return 1 if failures else 0

    print(f"Seeding {args.base}")
    seed(args.base, args.wipe)
    print("\nDone. Now verify:")
    print(f"  python scripts/seed_mockapi.py --base {args.base} --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
