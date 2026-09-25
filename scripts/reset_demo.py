#!/usr/bin/env python3
"""Put the demo back to its starting state so a run-through can be repeated.

    python scripts/reset_demo.py                            # everything
    python scripts/reset_demo.py --order BK-47110           # just the refund demo
    python scripts/reset_demo.py --oms $BASE                # also update mockapi
    python scripts/reset_demo.py --app http://127.0.0.1:8000

Three kinds of state get out of sync after a demo, and missing any one of them
leaves a run-through that fails halfway:

  1. Stripe   - the PaymentIntent behind BK-47110 has been fully refunded, and
                Stripe will not refund it twice. A NEW PaymentIntent is needed.
  2. the app  - returns, refunds, escalations, the audit log and chat sessions
                live in the server process. Restarting clears them; so does the
                reset endpoint, without losing your terminal.
  3. mockapi  - if the order records carry payment_intent_id, they now point at
                the spent payment.

Note the idempotency key below. seed_stripe.py uses `bookly-seed-{order_id}`,
so re-running IT returns the same (now empty) PaymentIntent rather than a fresh
one - correct for a seeder, useless for a reset. This script deliberately keys
each run uniquely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load as _load_env  # noqa: E402

_load_env()   # .env, so a new terminal tab is not a fresh start

import httpx  # noqa: E402

from app.backend import data  # noqa: E402
from app.backend.money import to_minor_units  # noqa: E402
from app.backend.payments import PAYMENT_MAP_PATH, STRIPE_API  # noqa: E402

TEST_PAYMENT_METHOD = "pm_card_visa"


def load_map() -> dict[str, str]:
    try:
        return json.loads(PAYMENT_MAP_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_map(m: dict[str, str]) -> None:
    PAYMENT_MAP_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def old_payment_state(c: httpx.Client, pi: str) -> str:
    """Report what happened to the payment we are replacing — the refunded
    amount is the receipt for the last demo run."""
    r = c.get(f"/payment_intents/{pi}")
    if r.status_code >= 400:
        return "not found"
    body = r.json()
    charges = (body.get("charges") or {}).get("data") or []
    refunded = charges[0].get("amount_refunded") if charges else None
    if refunded:
        return f"{body.get('status')}, £{refunded / 100:.2f} refunded"
    return str(body.get("status"))


def new_payments(key: str, orders: list[str]) -> dict[str, str]:
    mapping = load_map()
    run = uuid.uuid4().hex[:8]        # unique per reset, never reused

    with httpx.Client(base_url=STRIPE_API, timeout=30.0,
                      headers={"Authorization": f"Bearer {key}"}) as c:
        for order_id in orders:
            order = data.ORDERS.get(order_id)
            if not order:
                print(f"  ✗ {order_id}: not a Bookly order")
                continue

            previous = mapping.get(order_id)
            if previous:
                print(f"  · {order_id}: replacing {previous} ({old_payment_state(c, previous)})")

            r = c.post("/payment_intents", data={
                "amount": to_minor_units(order["total_gbp"]),
                "currency": "gbp",
                "payment_method": TEST_PAYMENT_METHOD,
                "confirm": "true",
                "automatic_payment_methods[enabled]": "true",
                "automatic_payment_methods[allow_redirects]": "never",
                "description": f"Bookly order {order_id} (demo reset {run})",
                "metadata[order_id]": order_id,
                "metadata[customer_id]": order["customer_id"],
                "metadata[reset_run]": run,
            }, headers={"Idempotency-Key": f"bookly-reset-{run}-{order_id}"})

            body = r.json()
            if r.status_code >= 400:
                err = body.get("error", {})
                print(f"  ✗ {order_id}: {err.get('code') or r.status_code} — "
                      f"{err.get('message','')[:150]}")
                continue
            if body.get("status") != "succeeded":
                print(f"  ✗ {order_id}: new PaymentIntent is '{body.get('status')}'")
                continue

            mapping[order_id] = body["id"]
            print(f"  ✓ {order_id}: fresh payment £{order['total_gbp']:.2f} → {body['id']}")

    save_map(mapping)
    return mapping


def reset_app(app_url: str) -> bool:
    """Clear the server's in-process ledgers and make it re-read the payment map."""
    try:
        r = httpx.post(f"{app_url.rstrip('/')}/api/admin/reset", timeout=10.0)
    except httpx.HTTPError as exc:
        print(f"  · app not reachable at {app_url} ({type(exc).__name__}) — "
              "restart it manually if it is running")
        return False

    if r.status_code == 404:
        print("  ✗ reset endpoint is disabled. Start the app with "
              "BOOKLY_DEMO_RESET=1 to enable it.")
        return False
    if r.status_code >= 400:
        print(f"  ✗ reset failed: {r.status_code} {r.text[:160]}")
        return False

    body = r.json()
    c = body["cleared"]
    print(f"  ✓ cleared {c['returns']} return(s), {c['refunds']} refund(s), "
          f"{c['escalations']} escalation(s), {c['sessions']} session(s)")
    print(f"  ✓ reloaded {body['payments_reloaded']} payment id(s)")
    return True


def push_to_oms(base: str, mapping: dict[str, str]) -> int:
    """Write the new payment ids onto the order records, and VERIFY.

    The order record is the system of record, and the gateway prefers it over
    the seeded map. That makes a failed write worse than no write at all: the
    OMS keeps pointing at the PaymentIntent the last run-through emptied, and
    it wins. So every update is read back, and anything that cannot be updated
    has the field cleared instead - which hands the decision to the map, where
    the fresh id already is.
    """
    problems = 0
    with httpx.Client(base_url=base.rstrip("/"), timeout=20.0) as c:
        try:
            rows = c.get("/orders").json() or []
        except (httpx.HTTPError, ValueError) as exc:
            print(f"  ✗ cannot read orders from {base} ({type(exc).__name__})")
            return 1

        for row in rows:
            order_id = row.get("order_id")
            pi = mapping.get(order_id)
            if not pi:
                continue

            current = row.get("payment_intent_id")
            if current == pi:
                print(f"  · {order_id}: already {pi}")
                continue

            r = c.put(f"/orders/{row['id']}", json={**row, "payment_intent_id": pi})

            if r.status_code < 400:
                # Verify. A 200 is not proof the field persisted.
                back = c.get("/orders", params={"order_id": order_id}).json() or []
                stored = (back[0].get("payment_intent_id") if back else None)
                if stored == pi:
                    print(f"  ✓ {order_id} ← {pi}")
                    continue
                print(f"  ! {order_id}: update accepted but stored '{stored}'")

            # The write did not stick. A stale id here beats the correct one in
            # the map, so remove it rather than leave a trap.
            cleared = c.put(f"/orders/{row['id']}", json={**row, "payment_intent_id": ""})
            if cleared.status_code < 400 and current:
                print(f"  ! {order_id}: could not store the new id "
                      f"({r.status_code}) - cleared the stale one, the seeded "
                      f"map will be used instead")
            elif current:
                print(f"  ✗ {order_id}: could not update OR clear "
                      f"payment_intent_id ({r.status_code}). The OMS still "
                      f"points at {current}, which the gateway prefers - "
                      f"refunds for this order will fail.")
                problems += 1
            else:
                print(f"  · {order_id}: OMS does not accept updates "
                      f"({r.status_code}); it carries no id, so the seeded map "
                      f"is used - this is fine")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", action="append", metavar="BK-#####",
                    help="reset only these orders (repeatable). "
                         "Default: every order with a payment.")
    ap.add_argument("--app", default="http://127.0.0.1:8000",
                    help="running app to clear (default %(default)s). "
                         "Pass --app '' to skip.")
    ap.add_argument("--oms", metavar="BASE_URL",
                    help="also write the new payment ids onto the mockapi orders")
    ap.add_argument("--skip-stripe", action="store_true",
                    help="only clear app state (use when PAYMENTS_BACKEND=ledger)")
    args = ap.parse_args()

    print("Resetting the Bookly demo\n")
    mapping = load_map()

    # 1. Stripe
    if not args.skip_stripe:
        key = os.environ.get("STRIPE_API_KEY", "")
        if not key:
            print("  · STRIPE_API_KEY not set — skipping Stripe "
                  "(fine if you are running PAYMENTS_BACKEND=ledger)")
        elif key.startswith("sk_live_"):
            print("  ✗ that is a LIVE key. Refusing.")
            return 1
        else:
            targets = args.order or sorted(mapping) or sorted(data.ORDERS)
            print("Stripe payments")
            mapping = new_payments(key, targets)
            print()

    # 2. mockapi order records
    #    Auto-detected: a reset that writes a new PaymentIntent but leaves the
    #    OMS pointing at the old one creates exactly the drift this script is
    #    supposed to remove.
    if not args.oms and os.environ.get("OMS_BASE_URL"):
        args.oms = os.environ["OMS_BASE_URL"]
        print(f"(using OMS_BASE_URL from the environment: {args.oms})\n")

    if args.oms:
        print("mockapi order records")
        push_to_oms(args.oms, mapping)
        print()

    # 3. the running app
    if args.app:
        print("App state")
        reset_app(args.app)
        print()

    print("Ready. Run through the demo again from the top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
