@echo off
echo ============================================
echo  EmotionAI — Setup ^& Launch
echo ============================================

IF NOT EXIST "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [2/3] Installing dependencies into venv...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\pip.exe install -r backend\requirements.txt --quiet

echo [3/3] Starting Flask backend...
cd backend
start "" ..\venv\Scripts\python.exe app.py
cd ..

echo.
echo Backend running at http://127.0.0.1:5000
echo Open frontend\index.html in your browser.
echo.
pause
