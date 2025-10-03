# Regenerate requirements.lock.txt from current venv
python -m pip freeze | Out-File -Encoding utf8 requirements.lock.txt
Write-Host "Wrote requirements.lock.txt"
