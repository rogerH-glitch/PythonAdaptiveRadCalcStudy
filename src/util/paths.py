from __future__ import annotations
from pathlib import Path


def get_outdir(outdir: str | Path) -> Path:
    """
    Normalize output directory without double-prefixing.
    - If caller passed 'results', we just use that path.
    - If caller passed an absolute or nested path, we use it verbatim.
    The directory is created if missing.
    """
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p
