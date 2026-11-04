@echo off
chcp 65001 >nul 2>&1
setlocal

REM ============================================================
REM  start-studio.bat  -  Launch LangGraph Studio
REM  Run it from the project root, or just double-click it.
REM  Press Ctrl+C to stop the server.
REM ============================================================

REM Switch to the directory this script lives in (the project root).
cd /d "%~dp0"

echo ============================================================
echo   Starting LangGraph Studio
echo ============================================================
echo.

REM --- Check 1: virtual environment ---
if not exist ".venv\Scripts\langgraph.exe" (
    echo [ERROR] .venv\Scripts\langgraph.exe not found.
    echo.
    echo Please make sure:
    echo   1. You are in the project root
    echo      ^(langgraph.json and pyproject.toml should be here^)
    echo   2. Dependencies are installed. If not, run:
    echo        uv sync
    echo        uv add --dev "langgraph-cli[inmem]"
    echo.
    pause
    exit /b 1
)

REM --- Check 2: .env ---
if not exist ".env" (
    echo [ERROR] .env not found.
    echo.
    echo Copy the template and fill in your DashScope key:
    echo        Copy-Item .env.example .env
    echo Then edit .env and replace DASHSCOPE_API_KEY with your real key.
    echo.
    pause
    exit /b 1
)

echo [1/2] Environment check passed
echo       Python : .venv\Scripts\python.exe
echo       Config : .env
echo.
echo [2/2] Starting the server, this may take a few seconds...
echo.
echo       When it is ready, open:
echo         https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
echo.
echo       Local API docs: http://127.0.0.1:2024/docs
echo.
echo       === Keep this window open. Press Ctrl+C to stop. ===
echo.

REM --allow-blocking is required: some langchain calls are synchronous
REM and would otherwise be rejected by the async server.
.venv\Scripts\langgraph.exe dev --port 2024 --allow-blocking

echo.
echo Server stopped.
pause
