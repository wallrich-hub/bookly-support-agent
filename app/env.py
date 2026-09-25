"""Load .env once, so configuration is set in one place instead of per terminal.

Environment variables are per-process, which means every new terminal tab is a
fresh start: the app knows your Stripe key, the reset script in the next tab
does not, and the resulting error blames the key rather than the tab. Reading a
.env file at import removes that entire class of confusion.

No dependency: this is a dozen lines and dotenv is not worth a requirement.

A variable already present in the real environment always wins. An explicit
`STRIPE_API_KEY=... python ...` must override the file, or overriding becomes
impossible and debugging gets worse rather than better.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_loaded = False


def load(path: Path | None = None) -> dict[str, str]:
    """Read KEY=value lines into os.environ without clobbering what is set."""
    global _loaded
    target = path or ENV_PATH
    applied: dict[str, str] = {}

    if not target.exists():
        _loaded = True
        return applied

    for raw in target.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue                      # the real environment wins
        os.environ[key] = value
        applied[key] = value

    _loaded = True
    return applied


def ensure_loaded() -> None:
    if not _loaded:
        load()
