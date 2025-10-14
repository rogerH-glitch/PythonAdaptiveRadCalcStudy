from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

def mod_to_file(mod: str) -> Path:
    """Map 'src.foo.bar' -> 'src/foo/bar.py' (simple heuristic)."""
    parts = mod.split(".")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    p = Path(*parts)  # e.g. src/foo/bar
    py = p.with_suffix(".py")
    init = p / "__init__.py"
    return py if py.exists() else init

def main():
    ap = argparse.ArgumentParser(description="Remove unreferenced modules identified by the audit (dry-run by default).")
    ap.add_argument("--audit", default="audit/legacy_report.json", help="Path to audit JSON")
    ap.add_argument("--apply", action="store_true", help="Actually delete files. Default is dry-run.")
    ap.add_argument("--trash", default="audit/trash", help="Directory to move deleted files (safety net)")
    args = ap.parse_args()

    report_path = Path(args.audit)
    if not report_path.exists():
        raise SystemExit(f"Audit file not found: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))
    unref: List[str] = data.get("unreferenced_modules", [])
    if not unref:
        print("No unreferenced modules listed in audit. Nothing to do.")
        return

    trash = Path(args.trash)
    trash.mkdir(parents=True, exist_ok=True)

    print("Planned removals (unreferenced modules):")
    to_delete = []
    for mod in unref:
        f = mod_to_file(mod)
        if f.exists():
            to_delete.append(f)
            print(f"  - {mod}  ->  {f}")
        else:
            print(f"  - {mod}  (no file found)")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to move files into:", trash)
        return

    # Move files to trash (soft delete)
    for f in to_delete:
        rel = f
        dest = trash / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
        f.unlink()
        print(f"Moved {f} -> {dest}")

    print("\nDone. Consider running tests:  pytest -q")

if __name__ == "__main__":
    main()
