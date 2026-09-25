#!/usr/bin/env python3
"""Create one succeeded Stripe test-mode payment per Bookly order.

    export STRIPE_API_KEY=sk_test_...
    python scripts/seed_stripe.py                       # create the payments
    python scripts/seed_stripe.py --check               # verify them
    python scripts/seed_stripe.py --push-oms $BASE      # write ids onto mockapi

Writes {order_id: payment_intent_id} to .stripe_payments.json, which the
StripeGateway reads when an order carries no payment_intent_id of its own.

Each PaymentIntent is created and confirmed in one call using Stripe's test
payment method, so it lands in `succeeded` and is immediately refundable.
Metadata carries order_id and customer_id, so a refund is traceable back to the
order from inside the Stripe dashboard - which is the thing worth showing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load as _load_env  # noqa: E402

_load_env()   # .env, so a new terminal tab is not a fresh start

import httpx  # noqa: E402

from app.backend import data  # noqa: E402
from app.backend.money import from_minor_units, to_minor_units  # noqa: E402
from app.backend.payments import PAYMENT_MAP_PATH, STRIPE_API  # noqa: E402

TEST_PAYMENT_METHOD = "pm_card_visa"   # Stripe's always-succeeds test card


def client(key: str) -> httpx.Client:
    return httpx.Client(base_url=STRIPE_API, timeout=30.0,
                        headers={"Authorization": f"Bearer {key}"})


def load_map() -> dict[str, str]:
    try:
        return json.loads(PAYMENT_MAP_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_map(m: dict[str, str]) -> None:
    PAYMENT_MAP_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def create_payments(key: str) -> dict[str, str]:
    mapping = load_map()

    with client(key) as c:
        for order_id, order in data.ORDERS.items():
            if order_id in mapping:
                print(f"  · {order_id}: already has {mapping[order_id]}, skipping")
                continue

            minor = to_minor_units(order["total_gbp"])
            form = {
                "amount": minor,
                "currency": "gbp",
                "payment_method": TEST_PAYMENT_METHOD,
                "confirm": "true",
                # No redirect-based methods: this is a server-side seed with no
                # browser to bounce through.
                "automatic_payment_methods[enabled]": "true",
                "automatic_payment_methods[allow_redirects]": "never",
                "description": f"Bookly order {order_id}",
                "metadata[order_id]": order_id,
                "metadata[customer_id]": order["customer_id"],
                "metadata[seeded_by]": "bookly-agent",
            }
            # Idempotent on the order id, so re-running never double-charges.
            r = c.post("/payment_intents", data=form,
                       headers={"Idempotency-Key": f"bookly-seed-{order_id}"})
            body = r.json()

            if r.status_code >= 400:
                err = body.get("error", {})
                print(f"  ✗ {order_id}: {err.get('code') or r.status_code} — "
                      f"{err.get('message', '')[:160]}")
                continue

            status = body.get("status")
            if status != "succeeded":
                print(f"  ✗ {order_id}: PaymentIntent is '{status}', not 'succeeded' "
                      "— it will not be refundable")
                continue

            mapping[order_id] = body["id"]
            print(f"  ✓ {order_id}: £{order['total_gbp']:.2f} → {body['id']}")

    save_map(mapping)
    print(f"\nWrote {PAYMENT_MAP_PATH.name} ({len(mapping)} payments)")
    return mapping


def check(key: str) -> int:
    """Confirm each payment exists, is succeeded, and has enough left to refund."""
    mapping = load_map()
    if not mapping:
        print("  ✗ no .stripe_payments.json — run without --check first")
        return 1

    problems = 0
    with client(key) as c:
        for order_id, order in data.ORDERS.items():
            pi = mapping.get(order_id)
            if not pi:
                print(f"  ✗ {order_id}: no payment on file")
                problems += 1
                continue

            r = c.get(f"/payment_intents/{pi}")
            if r.status_code >= 400:
                print(f"  ✗ {order_id}: {pi} not found ({r.status_code})")
                problems += 1
                continue

            body = r.json()
            expected = to_minor_units(order["total_gbp"])
            amount = body.get("amount")
            status = body.get("status")
            flags = []

            if status != "succeeded":
                flags.append(f"status={status}")
            if amount != expected:
                flags.append(f"amount={amount} expected={expected}")
            if body.get("livemode"):
                flags.append("LIVE MODE")

            # The item-level refund must fit inside what is left on the payment.
            item_total = to_minor_units(
                sum(i["price_gbp"] * i["qty"] for i in order["items"]))
            if item_total > amount:
                flags.append(f"items({item_total}) exceed payment({amount})")

            if flags:
                print(f"  ✗ {order_id}: {', '.join(flags)}")
                problems += 1
            else:
                print(f"  ✓ {order_id}: {pi} £{from_minor_units(amount):.2f} succeeded")

    return problems


def push_to_oms(base: str) -> int:
    """Write payment_intent_id onto the mockapi order records.

    A real OMS stores the payment reference on the order; the JSON map is a
    convenience for the in-memory path, not the shape you would ship.
    """
    mapping = load_map()
    problems = 0
    with httpx.Client(base_url=base.rstrip("/"), timeout=20.0) as c:
        rows = c.get("/orders").json() or []
        for row in rows:
            pi = mapping.get(row.get("order_id"))
            if not pi:
                continue
            r = c.put(f"/orders/{row['id']}", json={**row, "payment_intent_id": pi})
            if r.status_code >= 400:
                print(f"  ✗ {row['order_id']}: {r.status_code} {r.text[:120]}")
                problems += 1
            else:
                print(f"  ✓ {row['order_id']} ← {pi}")
    return problems


def _friendly(exc: Exception) -> int:
    """A connection failure is not a bug to debug, it is a thing to tell you."""
    print(f"\n  Could not reach Stripe at {STRIPE_API}")
    print(f"  {type(exc).__name__}: {str(exc)[:160]}")
    print("\n  Check your network, and that STRIPE_API_KEY is a live-looking "
          "test key (sk_test_...).")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, create nothing")
    ap.add_argument("--push-oms", metavar="BASE_URL",
                    help="also write the ids onto the mockapi order records")
    args = ap.parse_args()

    key = os.environ.get("STRIPE_API_KEY", "")
    if not key:
        print("Set STRIPE_API_KEY first:  export STRIPE_API_KEY=sk_test_...")
        return 1
    if key.startswith("sk_live_"):
        print("That is a LIVE key. Use a test key (sk_test_...) — refusing to continue.")
        return 1
    if not key.startswith("sk_test_"):
        print("Warning: key does not look like sk_test_... — continuing anyway.\n")

    if args.push_oms:
        print(f"Pushing payment ids to {args.push_oms}")
        return 1 if push_to_oms(args.push_oms) else 0

    if args.check:
        print("Checking Stripe payments\n")
        try:
            problems = check(key)
        except httpx.HTTPError as exc:
            return _friendly(exc)
        print("\nAll good." if not problems else f"\n{problems} problem(s).")
        return 1 if problems else 0

    print("Creating one test payment per order\n")
    try:
        create_payments(key)
    except httpx.HTTPError as exc:
        return _friendly(exc)
    print("\nNow verify:  python scripts/seed_stripe.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
