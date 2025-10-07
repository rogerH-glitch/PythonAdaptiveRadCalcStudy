from __future__ import annotations
from typing import Any, Tuple

_STORE: tuple[Any, Any, Any] | None = None

def capture(Y, Z, F) -> None:
    """Call this right after your grid evaluation produces Y, Z, F (same 2-D shape)."""
    global _STORE
    _STORE = (Y, Z, F)

def drain() -> Tuple[Any, Any, Any] | None:
    """Return the last captured (Y,Z,F) and clear the store."""
    global _STORE
    v = _STORE
    _STORE = None
    return v
