# One-liner: setup, tests, and run YAML suite with plots
.\scripts\setup.ps1
python -m pytest -q
python main.py --cases DOCS/validation_cases.yaml --outdir results --plot
Write-Host "Done. See ./results for CSV and plots."
