param([string]$Version = "1.0.0")

# 1) Run tests
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

# 2) Freeze deps
.\scripts\freeze-lock.ps1

# 3) Bump version in src/__init__.py if different
$init = "src/__init__.py"
(Get-Content $init) -replace '__version__ = ".*"', "__version__ = `"$Version`"" | Set-Content -Encoding UTF8 $init

git add -A
git commit -m "Release $Version"
git tag -a "v$Version" -m "Release $Version"
git push
git push --tags
Write-Host "Release v$Version pushed."
