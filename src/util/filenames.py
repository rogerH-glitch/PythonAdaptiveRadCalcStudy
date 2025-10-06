from __future__ import annotations
from datetime import datetime
from pathlib import Path


def ts_prefix(now: datetime | None = None) -> str:
    """
    Return a timestamp string 'yyyymmdd_HHMMSS' (e.g., '20251004_222628').
    (Your example used 20251004 which is yyyy-mm-dd -> yyyymmdd.)
    """
    dt = now or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def join_with_ts(outdir: str | Path, suffix: str, now: datetime | None = None) -> Path:
    """
    Build an output path with a timestamp prefix and a suffix you pass in, e.g.:
      join_with_ts('results', 'heatmap.png') -> results/20251004_222628_heatmap.png
    """
    p = Path(outdir)
    return p / f"{ts_prefix(now)}_{suffix}"
