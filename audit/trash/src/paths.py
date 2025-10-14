from __future__ import annotations
from pathlib import Path
import os

RESULTS_ROOT = Path("results")
TEST_SUBDIR = "test_results"

def resolve_outdir(outdir: str | None, *, test_run: bool = False) -> Path:
    """
    Resolve where to write outputs:
      - Base is 'results/' unless 'outdir' is absolute or explicitly nested.
      - If test_run=True (or running under pytest), append 'test_results/'.
      - Always mkdir(parents=True, exist_ok=True).
    """
    # Auto-detect pytest unless explicitly overridden
    if not test_run and "PYTEST_CURRENT_TEST" in os.environ:
        test_run = True

    if outdir:
        p = Path(outdir)
        if not p.is_absolute():
            p = RESULTS_ROOT / p
    else:
        p = RESULTS_ROOT

    if test_run:
        p = p / TEST_SUBDIR

    p.mkdir(parents=True, exist_ok=True)
    return p
