"""Single source of time.

Moving the OMS to a live API means order dates are real dates, so the policy
engine can no longer reason against a hardcoded constant. Everything that asks
"how long ago" now asks this module, which makes the return-window logic
testable (freeze it) and correct in production (don't).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

_FROZEN: datetime | None = None


def freeze(at: datetime) -> None:
    """Used by tests. Never called in the running app."""
    global _FROZEN
    _FROZEN = at


def unfreeze() -> None:
    global _FROZEN
    _FROZEN = None


def now() -> datetime:
    if _FROZEN is not None:
        return _FROZEN
    override = os.environ.get("BOOKLY_NOW")
    if override:
        return datetime.fromisoformat(override)
    return datetime.now(timezone.utc)


def days_since(iso: str) -> int:
    then = datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now() - then).days
