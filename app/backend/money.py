"""Money conversion.

Its own module with its own tests because this is where payment integrations
actually go wrong. `int(18.30 * 100)` is 1829, not 1830 - binary floating point
cannot represent 18.30, and the customer is short a penny with no error anywhere.

Every amount crossing into a payment processor goes through here.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

# Currencies with no minor unit (JPY, KRW...) take the amount as-is. GBP does
# not, but hardcoding x100 is the kind of assumption that breaks on expansion.
ZERO_DECIMAL = {"jpy", "krw", "vnd", "clp", "isk", "ugx", "xaf", "xof", "xpf"}


def to_minor_units(amount: float | str | Decimal, currency: str = "gbp") -> int:
    """£18.30 -> 1830. Decimal throughout; never float arithmetic."""
    value = Decimal(str(amount))
    if currency.lower() in ZERO_DECIMAL:
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    pence = (value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(pence)


def from_minor_units(minor: int, currency: str = "gbp") -> float:
    if currency.lower() in ZERO_DECIMAL:
        return float(minor)
    return float(Decimal(minor) / 100)


def refund_for_total(total: float | str | Decimal, currency: str = "gbp") -> Decimal:
    """The amount a cancellation refunds: the order total, quantized to the currency.

    Cancellation returns what the customer paid, not a figure the model names.
    `Decimal(str(total))` is deliberate. `str(22.50)` is `"22.5"`, and
    `Decimal("22.5")` is exact; `Decimal(22.50)` would already be the binary
    approximation. Quantizing then happens in decimal, never in float.
    """
    value = Decimal(str(total))
    if currency.lower() in ZERO_DECIMAL:
        return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
