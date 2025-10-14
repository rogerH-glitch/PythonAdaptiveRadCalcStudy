# Legacy Cleanup (Safe)

Use the audit output to remove unreferenced modules with a **dry-run first**. Files are moved to `audit/trash/` when you apply, so you can restore if needed.

## 1) Review audit

You should have:

- `audit/legacy_report.json`
- `audit/legacy_report.csv`

Open them and confirm the modules listed under `unreferenced_modules` are safe to remove.

## 2) Dry-run

```powershell
python -m tools.cleanup_legacy --audit audit/legacy_report.json
```

This prints the files that would be removed, without touching anything.

## 3) Apply (soft delete)

```powershell
python -m tools.cleanup_legacy --audit audit/legacy_report.json --apply
```

Files are moved to `audit/trash/…`. Run tests to confirm:

```powershell
pytest -q
```

If anything breaks, restore from `audit/trash/`.
