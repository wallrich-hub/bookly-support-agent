#!/usr/bin/env python3
"""Preflight: check every system the demo depends on, before you record.

    python scripts/doctor.py

Answers, in one screen, the question that costs the most time to answer any
other way: for each order, WHERE does the PaymentIntent come from, do the two
sources agree, and will Stripe actually accept a refund against it?

Reads only. Changes nothing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load as _load_env  # noqa: E402

_load_env()   # .env, so a new terminal tab is not a fresh start

import httpx  # noqa: E402

from app.backend import data  # noqa: E402
from app.backend.money import from_minor_units  # noqa: E402
from app.backend.payments import PAYMENT_MAP_PATH, STRIPE_API  # noqa: E402

OK, WARN, BAD, INFO = "  ✓", "  !", "  ✗", "  ·"
problems = 0


def flag(level: str, msg: str) -> None:
    global problems
    if level is BAD:
        problems += 1
    print(f"{level} {msg}")


def section(title: str) -> None:
    print(f"\n{title}\n" + "─" * len(title))


# --- 1. language model -----------------------------------------------------
def check_llm() -> None:
    section("Language model")
    from app.llm.providers import resolve_provider_name

    name = resolve_provider_name()
    if name == "mock":
        flag(WARN, "no API key found — running the rule-based planner, NOT an LLM")
        flag(INFO, "set ANTHROPIC_API_KEY (or OPENAI_API_KEY) to use a real model")
    else:
        flag(OK, f"{name}, model {os.environ.get('LLM_MODEL', 'default')}")


# --- 2. order source -------------------------------------------------------
def check_oms() -> dict[str, str]:
    """Returns {order_id: payment_intent_id} as the OMS reports it."""
    section("Order source")
    backend = os.environ.get("OMS_BACKEND", "memory").lower()
    oms_pis: dict[str, str] = {}

    if backend not in ("http", "mockapi"):
        flag(OK, "in-memory seed data (5 orders)")
        return oms_pis

    base = os.environ.get("OMS_BASE_URL", "")
    if not base:
        flag(BAD, "OMS_BACKEND=http but OMS_BASE_URL is not set")
        return oms_pis

    try:
        rows = httpx.get(f"{base.rstrip('/')}/orders", timeout=10.0).json() or []
    except (httpx.HTTPError, ValueError) as exc:
        flag(BAD, f"cannot reach {base} — {type(exc).__name__}")
        return oms_pis

    flag(OK, f"mockapi at {base} — {len(rows)} order record(s)")
    if len(rows) > len(data.ORDERS):
        flag(WARN, f"{len(rows)} records but only {len(data.ORDERS)} expected — "
                   "seeded twice? re-run seed_mockapi.py --wipe")
    for row in rows:
        if row.get("payment_intent_id"):
            oms_pis[row["order_id"]] = row["payment_intent_id"]
    flag(INFO, f"{len(oms_pis)} order(s) carry a payment_intent_id")
    return oms_pis


# --- 3. payments -----------------------------------------------------------
def check_payments(oms_pis: dict[str, str]) -> None:
    section("Payments")
    backend = os.environ.get("PAYMENTS_BACKEND", "ledger").lower()
    if backend != "stripe":
        flag(OK, "in-process ledger — refunds are recorded, no money moves")
        return

    key = os.environ.get("STRIPE_API_KEY", "")
    if not key:
        flag(BAD, "PAYMENTS_BACKEND=stripe but STRIPE_API_KEY is not set")
        return
    if key.startswith("sk_live_"):
        flag(BAD, "that is a LIVE key — the app will refuse to start")
        return

    mapped = {}
    if PAYMENT_MAP_PATH.exists():
        try:
            mapped = json.loads(PAYMENT_MAP_PATH.read_text())
        except ValueError:
            flag(BAD, f"{PAYMENT_MAP_PATH.name} is not valid JSON")
    else:
        flag(BAD, f"{PAYMENT_MAP_PATH.name} does not exist — "
                  "run: python scripts/seed_stripe.py")

    print()
    print("  order      source   payment            state")
    print("  ─────────  ───────  ─────────────────  ───────────────────────────")

    with httpx.Client(base_url=STRIPE_API, timeout=20.0,
                      headers={"Authorization": f"Bearer {key}"}) as c:
        for order_id, order in data.ORDERS.items():
            from_oms, from_map = oms_pis.get(order_id), mapped.get(order_id)
            # This mirrors the gateway exactly: the order record wins.
            pi = from_oms or from_map
            source = "OMS" if from_oms else ("map" if from_map else "—")

            if not pi:
                print(f"  {order_id}  {source:7}  {'none':17}  ", end="")
                flag(BAD, "no payment on file")
                continue

            try:
                r = c.get(f"/payment_intents/{pi}")
            except httpx.HTTPError as exc:
                print(f"  {order_id}  {source:7}  {pi[:17]:17}  ", end="")
                flag(BAD, f"Stripe unreachable ({type(exc).__name__})")
                continue

            if r.status_code >= 400:
                print(f"  {order_id}  {source:7}  {pi[:17]:17}  ", end="")
                flag(BAD, f"not found in Stripe ({r.status_code})")
                continue

            body = r.json()
            charges = (body.get("charges") or {}).get("data") or []
            refunded = charges[0].get("amount_refunded", 0) if charges else 0
            remaining = body.get("amount", 0) - refunded
            state = f"£{from_minor_units(remaining):.2f} refundable"

            issues = []
            if body.get("status") != "succeeded":
                issues.append(f"status={body.get('status')}")
            if body.get("livemode"):
                issues.append("LIVE MODE")
            if remaining <= 0:
                issues.append("fully refunded — run reset_demo.py")
            if from_oms and from_map and from_oms != from_map:
                issues.append(f"OMS and map disagree (map has {from_map[:14]}…)")

            print(f"  {order_id}  {source:7}  {pi[:17]:17}  ", end="")
            if issues:
                flag(BAD if remaining <= 0 else WARN, "; ".join(issues))
            else:
                flag(OK, state)


# --- 4. the running app ----------------------------------------------------
def check_app() -> None:
    section("Running app")
    url = os.environ.get("BOOKLY_APP_URL", "http://127.0.0.1:8000")
    try:
        body = httpx.get(f"{url}/api/health", timeout=5.0).json()
    except (httpx.HTTPError, ValueError):
        flag(INFO, f"not running at {url} (fine if you have not started it yet)")
        return

    flag(OK, f"up — provider={body['provider']}, oms={body['oms']}, "
             f"payments={body['payments']}")

    on_file = body.get("payments_on_file", 0)
    if body["payments"] == "stripe" and not on_file:
        flag(BAD, "the app holds NO payment ids. It was started before seeding — "
                  "restart it, or POST /api/admin/reset")
    elif body["payments"] == "stripe":
        flag(OK, f"{on_file} payment id(s) loaded")

    if not body.get("reset_enabled"):
        flag(INFO, "reset endpoint disabled — start with BOOKLY_DEMO_RESET=1 "
                   "to reset between takes")


def main() -> int:
    print("Bookly demo preflight")
    check_llm()
    oms_pis = check_oms()
    check_payments(oms_pis)
    check_app()

    print()
    if problems:
        print(f"{problems} problem(s) to fix before recording.")
    else:
        print("All clear.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
