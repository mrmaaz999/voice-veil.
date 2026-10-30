@echo off
setlocal
cd /d "%~dp0backend"
where python >nul 2>nul || (echo Python 3.10+ was not found. Install it from python.org and tick "Add Python to PATH". & pause & exit /b 1)
if not exist ..\.venv python -m venv ..\.venv
call ..\.venv\Scripts\activate
python -m pip install -q -r requirements.txt || (echo Dependency install failed. If it mentions a C++ compiler, install Python 3.12 from python.org. & pause & exit /b 1)
echo.
echo  Voice Veil starting. Open http://localhost:8000 in Chrome or Edge.
echo  First start trains the prototype model (about 20-60 s).
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
