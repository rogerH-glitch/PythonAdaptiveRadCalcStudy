param(
  [string]$Venv = ".venv",
  [switch]$Dev = $true
)

Write-Host "== Setup starting =="

# Use external venv if code sits under OneDrive to avoid .exe copy issues
$here = (Get-Location).Path
if ($here -like "*OneDrive*") {
  $Venv = "C:\venvs\parc"
  Write-Host "OneDrive detected → using external venv at $Venv"
}

if (!(Test-Path $Venv)) {
  Write-Host "Creating venv at $Venv"
  python -m venv $Venv
}

$activate = Join-Path $Venv "Scripts\Activate.ps1"
. $activate

python -m pip install -U pip setuptools wheel

if (Test-Path "requirements.lock.txt") {
  Write-Host "Installing from requirements.lock.txt"
  pip install -r requirements.lock.txt
} else {
  Write-Host "Installing from requirements.txt"
  pip install -r requirements.txt
  if ($Dev) {
    Write-Host "Installing dev requirements"
    pip install -r requirements-dev.txt
  }
  # create a lock snapshot for future machines
  Write-Host "Generating requirements.lock.txt"
  pip freeze | Out-File -Encoding utf8 requirements.lock.txt
}

# Quick sanity
python - << 'PY'
import sys
print("Python:", sys.executable)
try:
  import numpy, yaml, matplotlib
  print("numpy", numpy.__version__)
  print("pyyaml", yaml.__version__)
  print("matplotlib", matplotlib.__version__)
except Exception as e:
  print("Sanity import failed:", e)
PY

Write-Host "== Setup complete =="
