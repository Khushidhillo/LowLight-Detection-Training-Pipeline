@echo off
title Low-Light Detection Dashboard Launcher
echo ======================================================================
echo    LOW-LIGHT OBJECT DETECTION TRAINING PIPELINE SHOWCASE
echo    Zero-DCE++ Deep Curve Estimation x YOLOv8n (ExDark 12-Class)
echo ======================================================================
echo.
echo Checking if Dashboard Server is running on port 7860...
powershell -Command "if (Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue) { Write-Host 'Dashboard Server is already active on port 7860!' -ForegroundColor Green } else { Write-Host 'Starting Dashboard Server in background...' -ForegroundColor Yellow; Start-Process -FilePath '%~dp0.venv\Scripts\python.exe' -ArgumentList 'dashboard_server.py --port 7860' -WindowStyle Minimized }"

timeout /t 2 >nul
echo.
echo Opening Dashboard at http://localhost:7860 ...
start "" "http://localhost:7860"

echo.
echo ======================================================================
echo  READY!
echo  Your full motion research dashboard with live drag-and-drop
echo  enhancement and detection is now running at:
echo.
echo        http://localhost:7860
echo ======================================================================
pause
