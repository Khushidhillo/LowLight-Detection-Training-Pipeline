@echo off
echo ========================================================
echo  Launching Low-Light Detection Live Studio
echo  URL: http://localhost:7860
echo ========================================================
cd lowlight-training
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe dashboard_server.py --port 7860
) else (
    python dashboard_server.py --port 7860
)
pause
