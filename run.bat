@echo off
setlocal
echo Starting VisionGuard APDMS...
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py %*
) else (
    python main.py %*
)
endlocal
