# Start CycloneSense (API + UI)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$root;$root\backend"
$env:PYTHONUNBUFFERED = "1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$root`"; `$env:PYTHONPATH='$root;$root\backend'; python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$root\frontend`"; npm run dev"
Write-Host "API http://127.0.0.1:8000  UI http://localhost:5173"
