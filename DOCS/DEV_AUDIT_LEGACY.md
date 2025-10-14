# Dev Audit: Legacy Usage & Dead Modules

This utility helps find legacy patterns and possible dead code **without executing** the codebase.

## Run

```powershell
# From repo root
python -m tools.audit_legacy_usage --paths src tests --write
```

Outputs:

- Console summary (counts)
- `audit/legacy_report.json` – full detail (file, pattern, locations)
- `audit/legacy_report.csv` – quick triage list (category, file, detail)

## What it detects

- **Legacy geometry shapes**: `["xy"]["emitter"]`, `["xz"]["receiver"]`, `["corners3d"]`, and positional
  `build_display_geom(args, result)` calls.
- **Suspicious filenames**: names containing *legacy*, *old*, *original*, *backup*, *copy*, *tmp*, *deprecated*.
- **Static import graph** (AST-based): flags `src.*` modules that are **never imported** by other modules (heuristic).

> Note: *Unreferenced* doesn't mean *safe to delete* if something is reached dynamically (CLI entrypoints, plugin discovery, etc.). Use judgement.

## Triage workflow

1. **Open** `audit/legacy_report.csv` in Excel or VS Code.
2. Filter:
   - `category == legacy_pattern` → targets for refactor of geometry key shapes.
   - `category == build_display_geom_call` → places to migrate to the new API.
   - `category == suspect_filename` → rename/remove if unused.
   - `category == unreferenced_module` → candidates for deletion (confirm with grep/tests).
3. For any file you plan to remove:
   - `git grep <module_or_filename>` to confirm it's not referenced.
   - Run `pytest -q` after removal to verify.

## Future enhancements (optional)

- Integrate with `pre-commit` to block new legacy patterns.
- Emit `DeprecationWarning` from shims once the refactor lands.
